"""Unit tests for game.turns.TusTracker runtime event firing."""

import numpy as np
import pytest

from game.turns import TrackProfile, TusTracker, segment_turns
from tests.trackgen import stadium


@pytest.fixture
def profile():
    points, s, yaw, L = stadium(radius=100.0)
    turns = segment_turns(points, s, yaw, L)
    assert len(turns) == 2
    return TrackProfile(points, s, L, turns)


def drive(tracker, profile, start_idx, stop_idx, step_idx=2):
    """Feed centerline points [start_idx, stop_idx) to the tracker."""
    updates = []
    n = len(profile.points)
    for idx in range(start_idx, stop_idx, step_idx):
        x, y = profile.points[idx % n]
        updates.append(tracker.update(x, y))
    return updates


def all_beeps(updates):
    return [b for u in updates for b in u.beeps]


class TestForwardLap:
    def test_full_lap_event_sequence(self, profile):
        tracker = TusTracker(profile)
        # Drive slightly past the seam so a turn ending exactly at s = L
        # (arc-length 0) gets its TURN_END fired too.
        updates = drive(tracker, profile, 0, len(profile.points) + 16)

        beeps = all_beeps(updates)
        # Two turns, three approach beeps each, fired in order 1-2-3.
        assert [b.index for b in beeps] == [1, 2, 3, 1, 2, 3]
        assert all(b.direction == "right" for b in beeps)

        # Continuous beep turned on then off exactly twice.
        cont_flags = [u.continuous is not None for u in updates]
        transitions = np.flatnonzero(np.diff(np.asarray(cont_flags, dtype=int)))
        assert len(transitions) == 4

    def test_context_starts_at_first_beep_never_earlier(self, profile):
        tracker = TusTracker(profile)
        updates = drive(tracker, profile, 0, len(profile.points))

        first_beep_i = next(i for i, u in enumerate(updates) if u.beeps)
        first_ctx_i = next(i for i, u in enumerate(updates) if u.turn_context_active)
        # Intention preservation: context (and thus any gain shift) must not
        # lead the first beep.
        assert first_ctx_i == first_beep_i

    def test_context_clears_after_turn_end(self, profile):
        tracker = TusTracker(profile)
        updates = drive(tracker, profile, 0, len(profile.points))
        # Find the first cont-on stretch and check context drops with it.
        on = [u.continuous is not None for u in updates]
        first_on = on.index(True)
        first_off = on.index(False, first_on)
        assert updates[first_off].turn_context_active is False

    def test_second_lap_refires(self, profile):
        tracker = TusTracker(profile)
        n = len(profile.points)
        u1 = drive(tracker, profile, 0, n)
        # Wrap the seam and drive the first 100 m again (markers at ~140+ m
        # for turn 0 need 200 m to refire, so go to 220 m).
        u2 = drive(tracker, profile, 0, 220)
        assert len(all_beeps(u1)) == 6
        assert len(all_beeps(u2)) >= 3  # turn 0 approach beeps fired again


class TestRobustness:
    def test_reverse_and_recross_refires_beep(self, profile):
        tracker = TusTracker(profile)
        t0 = profile.turns[0]
        m1 = t0.marker_s[0]  # first approach marker arc-length
        i_before = int(m1 - 6)
        i_after = int(m1 + 6)

        drive(tracker, profile, 0, i_before)              # approach
        fwd = drive(tracker, profile, i_before, i_after)  # cross marker 1
        assert [b.index for b in all_beeps(fwd)] == [1]

        back = drive(tracker, profile, i_after, i_before, step_idx=-2)
        assert all_beeps(back) == []                      # reversing is silent

        fwd2 = drive(tracker, profile, i_before, i_after)
        assert [b.index for b in all_beeps(fwd2)] == [1]  # re-fires

    def test_teleport_resyncs_silently(self, profile):
        tracker = TusTracker(profile)
        drive(tracker, profile, 0, 10)
        x, y = profile.points[300]  # ~290 m jump >> MAX_PROGRESS_JUMP_M
        u = tracker.update(x, y)
        assert u.beeps == []
        assert u.continuous is None
        assert u.turn_context_active is False

    def test_teleport_mid_turn_kills_continuous(self, profile):
        tracker = TusTracker(profile)
        t0 = profile.turns[0]
        mid_turn_idx = int((t0.s_start + 20) % profile.track_length)
        drive(tracker, profile, 0, mid_turn_idx)
        assert tracker.cont is not None
        u = tracker.update(*profile.points[0])  # jump back to start
        assert u.continuous is None
        assert u.turn_context_active is False

    def test_off_track_freezes_state(self, profile):
        tracker = TusTracker(profile)
        updates = drive(tracker, profile, 0, 150)
        state_before = (tracker.cont, tracker.context_active, tracker.s_last)
        u = tracker.update(1e4, 1e4)
        assert u.beeps == []
        assert (tracker.cont, tracker.context_active, tracker.s_last) == state_before
