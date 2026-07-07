"""Offline rendering tests for game.audio.AudioManager (no audio device)."""

import numpy as np
import pytest

from game.audio import (
    MAX_ATTENUATION,
    TUS_BEEP_DURATION_S,
    TUS_FREQ_HZ,
    AudioManager,
)
from game.turns import BeepEvent, ContState, TusUpdate

FS = 44100
BLOCK = 512


def make_mgr(**kwargs):
    return AudioManager(sample_rate=FS, open_stream=False, **kwargs)


def render(mgr, blocks, updates=None):
    """Render `blocks` blocks, applying TusUpdates keyed by block index.

    Returns (signal (n, 2), slider_gains (n,), tus_gains (n,)).
    """
    out, sg, tg = [], [], []
    for i in range(blocks):
        if updates and i in updates:
            mgr.update_tus(updates[i])
        out.append(mgr.render_block(BLOCK))
        gains = mgr._dbg_last_gains
        sg.append(gains[0])
        tg.append(gains[1])
    return np.concatenate(out), np.concatenate(sg), np.concatenate(tg)


def dominant_freq(signal):
    spec = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    return np.fft.rfftfreq(len(signal), 1.0 / FS)[np.argmax(spec)]


def beep_update(direction="left", sharpness=2):
    return TusUpdate(beeps=[BeepEvent(1, direction, sharpness, 0)],
                     continuous=None, turn_context_active=True)


class TestBeeps:
    @pytest.mark.parametrize("sharpness", [0, 1, 2])
    def test_pitch_encodes_sharpness(self, sharpness):
        mgr = make_mgr()
        mgr.slider_vol = 0.0  # isolate the TUS channel
        sig, _, _ = render(mgr, 20, {2: beep_update(sharpness=sharpness)})
        freq = dominant_freq(sig[:, 0])
        assert freq == pytest.approx(TUS_FREQ_HZ[sharpness], abs=15.0)

    def test_left_turn_beeps_from_left(self):
        mgr = make_mgr()
        mgr.slider_vol = 0.0
        sig, _, _ = render(mgr, 20, {2: beep_update(direction="left")})
        assert np.abs(sig[:, 0]).max() > 0.01
        assert np.abs(sig[:, 1]).max() == 0.0  # hard pan, no bleed

    def test_right_turn_beeps_from_right(self):
        mgr = make_mgr()
        mgr.slider_vol = 0.0
        sig, _, _ = render(mgr, 20, {2: beep_update(direction="right")})
        assert np.abs(sig[:, 1]).max() > 0.01
        assert np.abs(sig[:, 0]).max() == 0.0

    def test_beep_length_matches_duration(self):
        mgr = make_mgr()
        mgr.slider_vol = 0.0
        sig, _, _ = render(mgr, 40, {0: beep_update()})
        nonzero = np.flatnonzero(np.abs(sig[:, 0]) > 1e-6)
        length_s = (nonzero[-1] - nonzero[0]) / FS
        assert length_s == pytest.approx(TUS_BEEP_DURATION_S, abs=0.01)


class TestContinuousBeep:
    def test_spans_turn_and_releases(self):
        mgr = make_mgr()
        mgr.slider_vol = 0.0
        on = TusUpdate([], ContState("right", 1, 0), True)
        off = TusUpdate([], None, False)
        sig, _, _ = render(mgr, 60, {5: on, 40: off})

        def block_rms(i):
            return float(np.sqrt(np.mean(sig[i * BLOCK:(i + 1) * BLOCK, 1] ** 2)))

        assert block_rms(2) == 0.0          # silent before
        assert block_rms(20) > 0.05          # sounding during the turn
        assert block_rms(55) < 1e-6          # released after
        assert dominant_freq(sig[10 * BLOCK:30 * BLOCK, 1]) == pytest.approx(
            TUS_FREQ_HZ[1], abs=15.0)

    def test_retrigger_changes_pitch_without_click(self):
        mgr = make_mgr()
        mgr.slider_vol = 0.0
        first = TusUpdate([], ContState("right", 1, 0), True)
        second = TusUpdate([], ContState("right", 2, 1), True)
        sig, _, _ = render(mgr, 80, {5: first, 40: second})
        assert dominant_freq(sig[15 * BLOCK:35 * BLOCK, 1]) == pytest.approx(
            TUS_FREQ_HZ[1], abs=15.0)
        assert dominant_freq(sig[45 * BLOCK:75 * BLOCK, 1]) == pytest.approx(
            TUS_FREQ_HZ[2], abs=15.0)
        # Click check: the largest sample-to-sample jump must stay within the
        # slew of an 880 Hz sine plus envelope movement.
        max_jump = np.max(np.abs(np.diff(sig[:, 1])))
        assert max_jump < 0.2


class TestAttentionalGains:
    def test_flat_mix_at_zero_strength(self):
        mgr = make_mgr(attentional_shift_strength=0.0)
        mgr.update_state(speed=10.0, ratio=0.5)
        _, sg, tg = render(mgr, 30, {5: beep_update()})
        assert np.all(sg == 1.0)
        assert np.all(tg == 1.0)

    def test_turn_context_ducks_slider(self):
        mgr = make_mgr(attentional_shift_strength=1.0)
        mgr.update_state(speed=10.0, ratio=0.5)
        _, sg, tg = render(mgr, 100, {10: beep_update()})
        assert sg[-1] == pytest.approx(1.0 - MAX_ATTENUATION)
        assert tg[-1] == pytest.approx(1.0)
        # Before the turn context: slider full, TUS attenuated.
        assert sg[0] == pytest.approx(1.0)
        assert tg[0] == pytest.approx(1.0 - MAX_ATTENUATION, abs=0.01)

    def test_gain_shift_never_precedes_first_beep(self):
        """Intention-preservation regression: the attentional transition must
        not begin in any audio block before the block carrying the first beep."""
        mgr = make_mgr(attentional_shift_strength=1.0)
        mgr.slider_vol = 0.0  # output is TUS-only, so onset is measurable
        beep_block = 12
        sig, sg, _ = render(mgr, 40, {beep_block: beep_update()})

        first_tus_sample = int(np.flatnonzero(np.abs(sig).sum(axis=1) > 0)[0])
        first_dev_sample = int(np.flatnonzero(sg < 1.0)[0])
        assert first_dev_sample // BLOCK == beep_block
        assert first_tus_sample // BLOCK == beep_block
        # No gain deviation in any earlier block.
        assert np.all(sg[:beep_block * BLOCK] == 1.0)


class TestLegacyEquivalence:
    def test_no_tus_events_means_pure_slider(self):
        """With no TUS traffic and zero shift, output equals the slider alone
        (the pre-Phase-2 sound)."""
        mgr_a = make_mgr(attentional_shift_strength=0.0)
        mgr_b = make_mgr(attentional_shift_strength=0.0)
        mgr_b.tus_vol = 0.0
        for m in (mgr_a, mgr_b):
            m.update_state(speed=12.0, ratio=0.3)
        sig_a, _, _ = render(mgr_a, 40)
        sig_b, _, _ = render(mgr_b, 40)
        assert np.array_equal(sig_a, sig_b)
        assert np.abs(sig_a).max() <= 0.3 + 1e-9  # legacy master volume bound
