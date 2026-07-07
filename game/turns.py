"""Track profiling and Turn Indicator System (TUS) runtime tracking.

Implements the RAD paper's turn indicator system (Smith & Nayar, CHI 2018)
for the CARLA racetrack:

* :func:`build_centerline` walks the map's waypoint graph once at startup to
  produce an ordered centerline (the all-lanes KD-tree in
  :class:`game.world.CarlaWorld` is an unordered point cloud and cannot be
  used for this).
* :func:`segment_turns` is a pure-NumPy function that classifies the
  centerline into straights and left/right turns of three sharpness classes
  (soft / moderate / sharp, thresholded in degrees of heading change per
  meter of track, per the paper) and places the four beep distance markers
  ahead of each turn.
* :class:`TrackProfile` bundles the centerline, the turn list, and a flat
  arc-length-sorted event table for cheap runtime lookups.
* :class:`TusTracker` converts the car's (x, y) position into TUS events
  each game-loop update: three discrete approach beeps, the continuous
  fourth beep spanning the turn, and the ``turn_context_active`` flag that
  drives the attentional channel rebalancing in
  :class:`game.audio.AudioManager`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# --- Track profiling constants ---
CENTERLINE_STEP_M = 1.0            # waypoint walk step when building the centerline
LOOP_CLOSURE_MIN_M = 200.0         # min distance before loop closure may trigger
MAX_CENTERLINE_M = 20_000.0        # hard cap; exceeded means the walk never closed

CURVATURE_SMOOTHING_WINDOW_M = 10.0
STRAIGHT_MAX_DEG_PER_M = 0.1       # below this the track counts as straight
SOFT_MAX_DEG_PER_M = 0.3           # paper: soft turns are < 0.3 deg/m
MODERATE_MAX_DEG_PER_M = 1.0       # paper: sharp turns are > 1 deg/m
MIN_TURN_LENGTH_M = 5.0            # turns shorter than this are noise
MERGE_GAP_M = 5.0                  # straight gaps shorter than this join adjacent turns
RIGHT_TURN_SIGN = +1               # CARLA/UE yaw is clockwise-positive from above

# --- TUS runtime constants ---
TUS_MARKER_SPACING_M = 20.0        # paper value (tuned for 35 m/s; rescale via CLI)
TUS_NUM_BEEPS = 4                  # 3 discrete approach beeps + 1 continuous at turn start
MAX_PROGRESS_JUMP_M = 30.0         # larger jumps are treated as teleports/resets
OFF_TRACK_SUPPRESS_M = 6.0         # freeze TUS when this far from the centerline
CONT_BEEP_ESCAPE_M = 10.0          # force the continuous beep off this far outside its turn

# Sharpness classes
SOFT, MODERATE, SHARP = 0, 1, 2

# Event kinds (order at equal arc-length: TURN_END fires before TURN_START so a
# split turn's continuous beep re-triggers with the new pitch, not stays off)
BEEP_1, BEEP_2, BEEP_3, TURN_START, TURN_END = range(5)
_KIND_ORDER = {TURN_END: 0, BEEP_1: 1, BEEP_2: 1, BEEP_3: 1, TURN_START: 2}


@dataclass
class TurnSegment:
    """One classified turn on the track, with its beep marker positions."""
    turn_id: int
    direction: str          # 'left' | 'right'
    sharpness: int          # SOFT | MODERATE | SHARP
    s_start: float
    s_end: float
    marker_s: list          # beep arc-lengths, last entry is always s_start
    is_continuation: bool   # True when split off the previous segment mid-turn
    approach_beep_nums: list = None  # beep number (1..3) per approach marker


@dataclass
class BeepEvent:
    """A single discrete approach beep to be played immediately."""
    index: int              # 1..3 (which approach marker fired)
    direction: str
    sharpness: int
    turn_id: int


@dataclass
class ContState:
    """State of the continuous fourth beep (playing while a turn is active)."""
    direction: str
    sharpness: int
    turn_id: int


@dataclass
class TusUpdate:
    """Result of one :meth:`TusTracker.update` call, consumed by the audio engine."""
    beeps: list             # BeepEvents fired this update (usually 0 or 1)
    continuous: object      # ContState | None
    turn_context_active: bool


def _wrap_deg(angle):
    """Wrap an angle in degrees to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def _in_interval(x, lo, hi, length):
    """True if arc-length ``x`` lies in [lo, hi) on a circular track."""
    if lo <= hi:
        return lo <= x < hi
    return x >= lo or x < hi


def build_centerline(carla_map, start_location, step=CENTERLINE_STEP_M):
    """Walk the waypoint graph from the spawn to produce an ordered centerline.

    One-time CARLA API use at startup. At junctions the walk prefers the
    branch on the same road and lane; otherwise it takes the branch with the
    smallest heading change, which keeps it on the Town04 highway loop
    rather than exiting onto a ramp.

    Parameters
    ----------
    carla_map : carla.Map
        The loaded map.
    start_location : carla.Location
        Any location on the racing lane (the fixed spawn point).
    step : float, optional
        Walk step in meters.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
        ``(points (N, 2), s (N,), yaw_deg (N,), track_length)`` where ``s``
        is the cumulative arc-length of each point and ``track_length``
        includes the closing gap back to the first point.

    Raises
    ------
    RuntimeError
        If the walk dead-ends or exceeds ``MAX_CENTERLINE_M`` without
        closing the loop.
    """
    import carla  # lazy import so the pure functions below stay testable without CARLA

    wp = carla_map.get_waypoint(
        start_location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    start_xy = np.array([wp.transform.location.x, wp.transform.location.y])

    pts, yaws = [], []
    total = 0.0
    cur = wp
    while True:
        loc = cur.transform.location
        pts.append((loc.x, loc.y))
        yaws.append(cur.transform.rotation.yaw)

        nxts = cur.next(step)
        if not nxts:
            raise RuntimeError(
                f"Centerline walk dead-ended after {total:.0f} m at ({loc.x:.1f}, {loc.y:.1f})."
            )
        if len(nxts) == 1:
            nxt = nxts[0]
        else:
            same = [w for w in nxts
                    if w.road_id == cur.road_id and w.lane_id == cur.lane_id]
            if same:
                nxt = same[0]
            else:
                cur_yaw = cur.transform.rotation.yaw
                nxt = min(nxts, key=lambda w: abs(
                    _wrap_deg(w.transform.rotation.yaw - cur_yaw)))

        nxt_loc = nxt.transform.location
        total += float(np.hypot(nxt_loc.x - loc.x, nxt_loc.y - loc.y))

        if total > LOOP_CLOSURE_MIN_M and np.hypot(
                nxt_loc.x - start_xy[0], nxt_loc.y - start_xy[1]) < step:
            break
        if total > MAX_CENTERLINE_M:
            raise RuntimeError(
                f"Centerline walk exceeded {MAX_CENTERLINE_M:.0f} m without loop "
                f"closure ({len(pts)} points). Check the junction branch heuristic."
            )
        cur = nxt

    points = np.asarray(pts, dtype=float)
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    track_length = float(s[-1] + np.hypot(*(points[0] - points[-1])))
    return points, s, np.asarray(yaws, dtype=float), track_length


def _runs(labels):
    """Split a label array into (start, stop_exclusive, label) runs."""
    boundaries = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(labels)]])
    return [(int(a), int(b), int(labels[a])) for a, b in zip(starts, stops)]


def segment_turns(points, s, yaw_deg, track_length, *,
                  marker_spacing=TUS_MARKER_SPACING_M,
                  smoothing_window_m=CURVATURE_SMOOTHING_WINDOW_M,
                  straight_max=STRAIGHT_MAX_DEG_PER_M,
                  soft_max=SOFT_MAX_DEG_PER_M,
                  moderate_max=MODERATE_MAX_DEG_PER_M,
                  min_turn_length_m=MIN_TURN_LENGTH_M,
                  merge_gap_m=MERGE_GAP_M):
    """Classify a closed centerline into turn segments and place beep markers.

    Pure NumPy — no CARLA dependency — so it is unit-testable with synthetic
    tracks. Curvature is the smoothed heading change per meter (deg/m);
    thresholds follow the RAD paper (soft < 0.3, sharp > 1.0) with an added
    straight floor below "soft". A turn whose sharpness class changes
    partway is split into separate segments sharing a boundary (the paper's
    behavior); the follow-on segment keeps only its ``s_start`` marker so the
    continuous beep re-triggers at the new pitch without spurious approach
    beeps.

    Returns
    -------
    list[TurnSegment]
        Ordered along the track starting from the longest straight.
    """
    n = len(points)
    if n < 3:
        return []

    # Per-sample curvature (deg/m) between consecutive points, circular.
    nxt = np.roll(np.arange(n), -1)
    ds = np.hypot(*(points[nxt] - points).T)
    ds[-1] = max(ds[-1], 1e-9)  # closing gap; guard zero
    ds = np.maximum(ds, 1e-9)
    dyaw = _wrap_deg(yaw_deg[nxt] - yaw_deg)
    kappa = dyaw / ds

    # Circular moving-average smoothing.
    window = max(1, int(round(smoothing_window_m / max(float(np.median(ds)), 1e-9))))
    if window > 1:
        pad = window // 2
        padded = np.concatenate([kappa[-pad:], kappa, kappa[:pad]])
        kernel = np.ones(window) / window
        kappa = np.convolve(padded, kernel, mode="same")[pad:pad + n]

    # Per-sample labels: 0 = straight, +1 = right turn, -1 = left turn.
    # Samples spanning an outsized gap (e.g. an imperfectly closed loop's
    # seam) carry no usable curvature; treat them as straight connectors.
    connector = ds > 5.0 * float(np.median(ds))
    abs_k = np.abs(kappa)
    is_turn = (abs_k >= straight_max) & ~connector
    dir_label = np.where(is_turn, np.sign(kappa * RIGHT_TURN_SIGN).astype(int), 0)
    sharp_label = np.where(abs_k < soft_max, SOFT,
                           np.where(abs_k < moderate_max, MODERATE, SHARP))

    if not np.any(dir_label == 0):
        rot = 0  # no straight anywhere; degenerate but proceed unrotated
    else:
        # Rotate so index 0 sits inside the longest straight run (removes all
        # wraparound special-casing from the run logic below).
        straight = np.concatenate([dir_label == 0, dir_label == 0])
        best_len, best_start, run = 0, 0, 0
        for i, flag in enumerate(straight):
            run = run + 1 if flag else 0
            if run > best_len and run <= n:
                best_len, best_start = run, i - run + 1
        rot = (best_start + best_len // 2) % n

    d_rot = np.roll(dir_label, -rot)
    k_rot = np.roll(sharp_label, -rot)
    orig_idx = (np.arange(n) + rot) % n

    def run_length(a, b):
        lo = s[orig_idx[a]]
        hi = s[orig_idx[b]] if b < n else s[orig_idx[0]]
        return (hi - lo) % track_length if track_length > 0 else 0.0

    # Fill short straight gaps between same-direction turn runs.
    runs = _runs(d_rot)
    for j in range(1, len(runs) - 1):
        a, b, label = runs[j]
        if label != 0:
            continue
        _, _, prev_label = runs[j - 1]
        _, _, next_label = runs[j + 1]
        if prev_label == next_label and prev_label != 0 and run_length(a, b) < merge_gap_m:
            d_rot[a:b] = prev_label
            k_rot[a:b] = k_rot[a - 1]

    # Drop turn runs that are too short to be real turns.
    for a, b, label in _runs(d_rot):
        if label != 0 and run_length(a, b) < min_turn_length_m:
            d_rot[a:b] = 0

    # Build segments: one per (direction run x sharpness class), splitting on
    # class changes and absorbing sub-runs too short to stand alone.
    segments = []
    for a, b, label in _runs(d_rot):
        if label == 0:
            continue
        sub = _runs(k_rot[a:b])
        sub = [(a + sa, a + sb, cls) for sa, sb, cls in sub]
        # Merge short sharpness sub-runs into their longer neighbor. The
        # smoothing window creates transition-class bands up to one window
        # wide at every entry/exit, so anything shorter than the window is
        # an artifact, not a real class change.
        sub_min = max(min_turn_length_m, smoothing_window_m)
        changed = True
        while changed and len(sub) > 1:
            changed = False
            lengths = [run_length(sa, sb) for sa, sb, _ in sub]
            j = int(np.argmin(lengths))
            if lengths[j] < sub_min:
                if j == 0:
                    merged = (sub[0][0], sub[1][1], sub[1][2])
                    sub = [merged] + sub[2:]
                elif j == len(sub) - 1:
                    merged = (sub[-2][0], sub[-1][1], sub[-2][2])
                    sub = sub[:-2] + [merged]
                else:
                    left_len, right_len = lengths[j - 1], lengths[j + 1]
                    if left_len >= right_len:
                        merged = (sub[j - 1][0], sub[j][1], sub[j - 1][2])
                        sub = sub[:j - 1] + [merged] + sub[j + 1:]
                    else:
                        merged = (sub[j][0], sub[j + 1][1], sub[j + 1][2])
                        sub = sub[:j] + [merged] + sub[j + 2:]
                changed = True
        direction = "right" if label > 0 else "left"
        for m, (sa, sb, cls) in enumerate(sub):
            s_start = float(s[orig_idx[sa]])
            s_end = float(s[orig_idx[sb]]) if sb < n else float(s[orig_idx[0]])
            segments.append(TurnSegment(
                turn_id=len(segments),
                direction=direction,
                sharpness=int(cls),
                s_start=s_start,
                s_end=s_end,
                marker_s=[],
                is_continuation=(m > 0),
            ))

    # Order along the track (rotation made the order start from an arbitrary
    # straight) and renumber so turn 0 is the first turn past arc-length 0.
    segments.sort(key=lambda t: t.s_start)
    for i, t in enumerate(segments):
        t.turn_id = i

    # Place beep markers; clamp approach markers that land inside any turn.
    for seg_ in segments:
        if seg_.is_continuation:
            seg_.marker_s = [seg_.s_start]
            seg_.approach_beep_nums = []
            continue
        markers, nums = [], []
        for k in range(TUS_NUM_BEEPS - 1, 0, -1):
            m_s = (seg_.s_start - k * marker_spacing) % track_length
            inside_a_turn = any(
                _in_interval(m_s, other.s_start, other.s_end, track_length)
                for other in segments
            )
            if not inside_a_turn:
                markers.append(m_s)
                nums.append(TUS_NUM_BEEPS - k)  # k=3 -> beep 1, k=1 -> beep 3
        markers.append(seg_.s_start)
        seg_.marker_s = markers
        seg_.approach_beep_nums = nums

    return segments


class TrackProfile:
    """Immutable per-track data: centerline, turn list, and the event table.

    Built once in :class:`game.app.RacingApp` after the world is up. Owns its
    own KD-tree over the centerline points — deliberately not the world's
    all-lanes ``waypoint_tree``, which indexes every lane in both directions.
    """

    def __init__(self, points, s, track_length, turns):
        self.points = points
        self.s = s
        self.track_length = track_length
        self.turns = turns
        self.turns_by_id = {t.turn_id: t for t in turns}
        self.tree = cKDTree(points)

        events = []
        for t in turns:
            nums = t.approach_beep_nums or []
            for m_s, num in zip(t.marker_s[:-1], nums):
                events.append((m_s, BEEP_1 + num - 1, t.turn_id))
            events.append((t.marker_s[-1], TURN_START, t.turn_id))
            events.append((t.s_end, TURN_END, t.turn_id))
        if events:
            ev_s = np.array([e[0] for e in events])
            ev_kind = np.array([e[1] for e in events])
            ev_turn = np.array([e[2] for e in events])
            order = np.lexsort((np.array([_KIND_ORDER[k] for k in ev_kind]), ev_s))
            self.event_s = ev_s[order]
            self.event_kind = ev_kind[order]
            self.event_turn = ev_turn[order]
        else:
            self.event_s = np.empty(0)
            self.event_kind = np.empty(0, dtype=int)
            self.event_turn = np.empty(0, dtype=int)

    @classmethod
    def build(cls, carla_map, start_location, *,
              step=CENTERLINE_STEP_M, marker_spacing=TUS_MARKER_SPACING_M):
        """Build the full profile from a live CARLA map (startup only)."""
        points, s, yaw, track_length = build_centerline(carla_map, start_location, step)
        turns = segment_turns(points, s, yaw, track_length, marker_spacing=marker_spacing)
        return cls(points, s, track_length, turns)

    def summary(self):
        """Human-readable turn table for the startup log / manual test checklist."""
        names = {SOFT: "soft", MODERATE: "moderate", SHARP: "sharp"}
        lines = [f"Track profile: {self.track_length:.0f} m, {len(self.turns)} turn segments"]
        for t in self.turns:
            cont = " (continuation)" if t.is_continuation else ""
            lines.append(
                f"  turn {t.turn_id:2d}: {t.direction:5s} {names[t.sharpness]:8s} "
                f"s=[{t.s_start:7.1f}, {t.s_end:7.1f}] "
                f"beeps at {[round(m, 1) for m in t.marker_s]}{cont}"
            )
        return "\n".join(lines)


class TusTracker:
    """Converts car positions into TUS events by arc-length interval membership.

    Stateless with respect to any event pointer: each update binary-searches
    the profile's sorted event table for events whose arc-length falls in
    ``(s_last, s_car]``, so reversing and re-crossing a marker simply
    re-fires it and there is no pointer state to corrupt.
    """

    def __init__(self, profile):
        self.profile = profile
        self.s_last = None
        self.cont = None
        self.context_active = False

    def _frozen(self):
        return TusUpdate([], self.cont, self.context_active)

    def _events_between(self, lo, hi):
        """Indices of events with s in (lo, hi], handling wraparound."""
        ev_s = self.profile.event_s
        if len(ev_s) == 0:
            return []
        a = np.searchsorted(ev_s, lo, side="right")
        b = np.searchsorted(ev_s, hi, side="right")
        if lo <= hi:
            return list(range(a, b))
        return list(range(a, len(ev_s))) + list(range(0, b))

    def _check_escape(self, s_car):
        """Force the continuous beep off if the car left its turn by a margin."""
        if self.cont is None:
            return
        turn = self.profile.turns_by_id[self.cont.turn_id]
        L = self.profile.track_length
        if _in_interval(s_car, turn.s_start, turn.s_end, L):
            return
        dist_past_end = (s_car - turn.s_end) % L
        dist_before_start = (turn.s_start - s_car) % L
        if min(dist_past_end, dist_before_start) > CONT_BEEP_ESCAPE_M:
            self.cont = None
            self.context_active = False

    def update(self, x, y):
        """Advance the tracker to the car's current position.

        Returns
        -------
        TusUpdate
            Beeps fired this step, the continuous-beep state, and the
            attentional context flag. ``turn_context_active`` becomes True in
            the same update that fires a turn's first beep — never earlier —
            which is what guarantees the attentional gain shift cannot act as
            an early-warning cue (intention preservation).
        """
        dist, idx = self.profile.tree.query([x, y])
        if dist > OFF_TRACK_SUPPRESS_M:
            return self._frozen()

        s_car = float(self.profile.s[idx])
        if self.s_last is None:
            self.s_last = s_car
            return self._frozen()

        L = self.profile.track_length
        delta = ((s_car - self.s_last + L / 2.0) % L) - L / 2.0

        if abs(delta) > MAX_PROGRESS_JUMP_M:
            # Teleport / reset: resync silently.
            self.s_last = s_car
            self.cont = None
            self.context_active = False
            return TusUpdate([], None, False)

        if delta <= 0:
            self.s_last = s_car
            self._check_escape(s_car)
            return self._frozen()

        beeps = []
        for i in self._events_between(self.s_last, s_car):
            kind = int(self.profile.event_kind[i])
            turn = self.profile.turns_by_id[int(self.profile.event_turn[i])]
            if kind in (BEEP_1, BEEP_2, BEEP_3):
                beeps.append(BeepEvent(kind - BEEP_1 + 1, turn.direction,
                                       turn.sharpness, turn.turn_id))
                self.context_active = True
            elif kind == TURN_START:
                self.cont = ContState(turn.direction, turn.sharpness, turn.turn_id)
                self.context_active = True
            elif kind == TURN_END:
                if self.cont is not None and self.cont.turn_id == turn.turn_id:
                    self.cont = None
                    self.context_active = False

        self.s_last = s_car
        self._check_escape(s_car)
        return TusUpdate(beeps, self.cont, self.context_active)
