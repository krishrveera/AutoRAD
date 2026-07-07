import collections

import numpy as np
import sounddevice as sd

# --- Channel volumes ---
SLIDER_VOL = 0.3        # master volume of the sound-slider engine tone
TUS_VOL = 0.25          # master volume of the turn-indicator channel
TUS_PAN_BLEED = 0.0     # bleed of TUS beeps into the opposite ear (0 = hard pan)

# --- TUS beep synthesis ---
TUS_FREQ_HZ = {0: 440.0, 1: 660.0, 2: 880.0}  # soft / moderate / sharp pitch
TUS_BEEP_DURATION_S = 0.15
TUS_BEEP_ATTACK_S = 0.010
TUS_BEEP_RELEASE_S = 0.030
TUS_CONT_ATTACK_S = 0.020
TUS_CONT_RELEASE_S = 0.030
BEEP_CUT_FADE_S = 0.005  # fade applied when a new beep cuts a playing one

# --- Attentional channel rebalancing ---
MAX_ATTENUATION = 0.5    # gain reduction of the de-emphasized channel at shift = 1
ATTENTION_RAMP_S = 0.4   # seconds for a full 0-to-1 gain transition


class AudioManager:
    """Real-time stereo synthesizer for both RAD audio channels.

    Channel 1 — the *sound slider*: a sawtooth "engine" tone (pitch rising
    with speed) panned across the stereo field by the RAD ratio from
    :class:`~game.ego.Ego`, with ``warp_factor`` exponentially warping the
    pan curve to widen a centered "safe zone".

    Channel 2 — the *turn indicator system (TUS)*: discrete approach beeps
    and a continuous fourth beep spanning each turn, fed by
    :class:`~game.turns.TusTracker` via :meth:`update_tus`. Beep pitch
    encodes turn sharpness and stereo side encodes turn direction, per the
    RAD paper.

    An attentional gain stage rebalances the two channels by
    ``attentional_shift_strength``: on a straightaway the slider is dominant
    and the TUS attenuated; while a turn is active the emphasis flips. The
    transition ramps over ``ATTENTION_RAMP_S`` and — critically — is driven
    by the same update that fires a turn's first beep, so the gain shift can
    never precede the beep and act as an implicit early warning (intention
    preservation).

    All DSP lives in :meth:`render_block` so tests can render offline with
    ``open_stream=False``.
    """

    def __init__(self, sample_rate=44100, warp_factor=2.0,
                 attentional_shift_strength=0.0, open_stream=True):
        """Create the stereo output stream and initialize synth state.

        Parameters
        ----------
        sample_rate : int, optional
            Audio sample rate in Hz.
        warp_factor : float, optional
            Exponent applied to the pan value. ``1.0`` is linear (no safe
            zone), ``2.0`` is the standard squared safe zone, and ``3.0+``
            produces an extreme safe zone with violent edge warnings.
        attentional_shift_strength : float, optional
            0.0 keeps both channels at a constant relative mix (flat, the
            pre-Phase-2 behavior); 1.0 applies the full contextual emphasis
            swing between the slider and TUS channels.
        open_stream : bool, optional
            Set False in tests to synthesize without an audio device.
        """
        self.fs = sample_rate
        self.stream = (
            sd.OutputStream(samplerate=self.fs, channels=2, callback=self._callback)
            if open_stream else None
        )

        # State variables updated by the Ego class
        self.speed = 0.0
        self.ratio = 0.5
        self.is_running = False

        # Internal phase tracking to prevent 'clicks' between buffers
        self.phase = 0.0

        # Store the parameter in the class state
        # 1.0 = Linear (No safe zone)
        # 2.0 = Squared (Standard safe zone)
        # 3.0+ = Extreme (Massive safe zone, violent edge warnings)
        self.warp_factor = warp_factor

        # Channel volumes as instance state so tests can mute one channel.
        self.slider_vol = SLIDER_VOL
        self.tus_vol = TUS_VOL

        # --- TUS state written by the game thread (GIL-atomic rebinds) ---
        self.beep_queue = collections.deque(maxlen=8)
        self.cont_beep = None            # ContState | None
        self.turn_context_active = False

        # --- Synth state owned by the audio thread ---
        self._beep = None                # active discrete beep
        self._beep_fading = None         # beep being cut by a newer one
        self._cont = {"freq": TUS_FREQ_HZ[0], "phase": 0.0, "level": 0.0,
                      "left": 1.0, "right": 1.0}
        self.attentional_shift_strength = attentional_shift_strength
        self._slider_gain = 1.0
        self._tus_gain = 1.0 - attentional_shift_strength * MAX_ATTENUATION

    # ------------------------------------------------------------------
    # Game-thread interface
    # ------------------------------------------------------------------

    def update_state(self, speed, ratio):
        """Called by your main loop to feed simulation data to the audio thread."""
        self.speed = speed
        self.ratio = ratio

    def update_tus(self, update):
        """Feed a :class:`~game.turns.TusUpdate` from the game loop.

        Appends fired beeps to the queue and rebinds the continuous-beep and
        context flags; the audio callback consumes all three at the top of
        its next block, which keeps the beep onset and any attentional gain
        shift within the same buffer.
        """
        for beep in update.beeps:
            self.beep_queue.append(beep)
        self.cont_beep = update.continuous
        self.turn_context_active = update.turn_context_active

    # ------------------------------------------------------------------
    # Audio-thread DSP
    # ------------------------------------------------------------------

    @staticmethod
    def _pan(direction):
        """Stereo gains for a TUS sound: hard-panned to the turn's side."""
        if direction == "left":
            return 1.0, TUS_PAN_BLEED
        return TUS_PAN_BLEED, 1.0

    def _ramp(self, current, target, frames):
        """Per-sample linear ramp of a gain value toward its target."""
        if current == target:
            return np.full(frames, current), current
        step = 1.0 / (ATTENTION_RAMP_S * self.fs)
        delta = np.minimum(np.arange(1, frames + 1) * step, abs(target - current))
        vec = current + np.sign(target - current) * delta
        return vec, float(vec[-1])

    def _render_slider(self, frames):
        """Sawtooth engine tone with warped equal-power panning (channel 1)."""
        freq = 50 + (self.speed * 5.0)
        t = np.arange(frames) / self.fs
        cycles = freq * t + self.phase
        wave = 2 * (cycles - np.floor(0.5 + cycles))
        self.phase = cycles[-1] % 1.0

        rad_clamped = max(0.0, min(1.0, self.ratio))
        pan_x = (rad_clamped - 0.5) * 2.0
        pan_warped = np.sign(pan_x) * (np.abs(pan_x) ** self.warp_factor)
        rad_warped = (pan_warped + 1.0) / 2.0

        left_gain = np.cos((np.pi / 2.0) * rad_warped)
        right_gain = np.sin((np.pi / 2.0) * rad_warped)
        return wave, left_gain, right_gain

    def _beep_envelope(self, positions, total):
        """Raised-cosine attack/release envelope for a discrete beep."""
        att = max(1, int(TUS_BEEP_ATTACK_S * self.fs))
        rel = max(1, int(TUS_BEEP_RELEASE_S * self.fs))
        attack = 0.5 - 0.5 * np.cos(np.pi * np.clip(positions / att, 0.0, 1.0))
        release = 0.5 - 0.5 * np.cos(np.pi * np.clip((total - positions) / rel, 0.0, 1.0))
        return np.minimum(attack, release)

    def _render_tus(self, frames):
        """Discrete + continuous TUS beeps (channel 2). Returns (L, R) arrays."""
        left = np.zeros(frames)
        right = np.zeros(frames)

        # Consume newly fired beeps; the newest wins, cutting any active one.
        while self.beep_queue:
            beep = self.beep_queue.popleft()
            if self._beep is not None:
                self._beep_fading = {
                    **self._beep,
                    "fade_total": max(1, int(BEEP_CUT_FADE_S * self.fs)),
                    "fade_pos": 0,
                }
            self._beep = {
                "freq": TUS_FREQ_HZ[beep.sharpness],
                "pos": 0,
                "total": int(TUS_BEEP_DURATION_S * self.fs),
                "pan": self._pan(beep.direction),
            }

        # Active discrete beep.
        if self._beep is not None:
            b = self._beep
            n = min(frames, b["total"] - b["pos"])
            pos = b["pos"] + np.arange(n)
            wave = np.sin(2.0 * np.pi * b["freq"] * pos / self.fs)
            wave *= self._beep_envelope(pos, b["total"])
            left[:n] += wave * b["pan"][0]
            right[:n] += wave * b["pan"][1]
            b["pos"] += n
            if b["pos"] >= b["total"]:
                self._beep = None

        # Tail of a beep that was cut by a newer one: quick linear fade.
        if self._beep_fading is not None:
            f = self._beep_fading
            n = min(frames, f["fade_total"] - f["fade_pos"])
            pos = f["pos"] + np.arange(n)
            wave = np.sin(2.0 * np.pi * f["freq"] * pos / self.fs)
            wave *= self._beep_envelope(pos, f["total"])
            fade = 1.0 - (f["fade_pos"] + np.arange(1, n + 1)) / f["fade_total"]
            wave *= np.maximum(fade, 0.0)
            left[:n] += wave * f["pan"][0]
            right[:n] += wave * f["pan"][1]
            f["pos"] += n
            f["fade_pos"] += n
            if f["fade_pos"] >= f["fade_total"] or f["pos"] >= f["total"]:
                self._beep_fading = None

        # Continuous fourth beep: sustained tone with its own phase
        # accumulator; frequency jumps (phase-continuously) when a split
        # turn re-triggers it at a new sharpness.
        cont_state = self.cont_beep  # atomic read of the game thread's rebind
        c = self._cont
        if cont_state is not None:
            c["freq"] = TUS_FREQ_HZ[cont_state.sharpness]
            c["left"], c["right"] = self._pan(cont_state.direction)
        target = 1.0 if cont_state is not None else 0.0
        if c["level"] > 0.0 or target > 0.0:
            slope = (1.0 / (TUS_CONT_ATTACK_S * self.fs) if target > 0.0
                     else -1.0 / (TUS_CONT_RELEASE_S * self.fs))
            env = np.clip(c["level"] + slope * np.arange(1, frames + 1), 0.0, 1.0)
            c["level"] = float(env[-1])
            inc = c["freq"] / self.fs
            cycles = c["phase"] + inc * np.arange(1, frames + 1)
            wave = np.sin(2.0 * np.pi * cycles) * env
            c["phase"] = cycles[-1] % 1.0
            left += wave * c["left"]
            right += wave * c["right"]

        return left, right

    def render_block(self, frames):
        """Synthesize one stereo block; all DSP happens here (testable offline)."""
        out = np.empty((frames, 2))

        # Attentional gain targets from the context flag set by update_tus.
        shift = self.attentional_shift_strength * MAX_ATTENUATION
        if self.turn_context_active:
            slider_target, tus_target = 1.0 - shift, 1.0
        else:
            slider_target, tus_target = 1.0, 1.0 - shift
        slider_gain, self._slider_gain = self._ramp(self._slider_gain, slider_target, frames)
        tus_gain, self._tus_gain = self._ramp(self._tus_gain, tus_target, frames)

        wave, left_gain, right_gain = self._render_slider(frames)
        tus_left, tus_right = self._render_tus(frames)

        out[:, 0] = wave * left_gain * self.slider_vol * slider_gain \
            + tus_left * self.tus_vol * tus_gain
        out[:, 1] = wave * right_gain * self.slider_vol * slider_gain \
            + tus_right * self.tus_vol * tus_gain

        # Exposed for the offline regression tests (gain-vs-beep ordering).
        self._dbg_last_gains = (slider_gain, tus_gain)
        return out

    def _callback(self, outdata, frames, time, status):
        """This is the high-priority thread that generates audio buffers."""
        outdata[:] = self.render_block(frames)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Begin audio playback by starting the output stream."""
        if self.stream is not None:
            self.stream.start()
        self.is_running = True

    def stop(self):
        """Halt audio playback by stopping the output stream."""
        if self.stream is not None:
            self.stream.stop()
        self.is_running = False
