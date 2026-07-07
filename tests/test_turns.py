"""Unit tests for game.turns.segment_turns on synthetic tracks."""

import numpy as np
import pytest

from game.turns import (
    MODERATE,
    SHARP,
    SOFT,
    TUS_MARKER_SPACING_M,
    segment_turns,
)
from tests.trackgen import build_track, stadium


def turn_length(seg, track_length):
    return (seg.s_end - seg.s_start) % track_length


class TestClassification:
    @pytest.mark.parametrize("radius,expected", [
        (250.0, SOFT),      # 180/(pi*250) = 0.23 deg/m
        (100.0, MODERATE),  # 0.57 deg/m
        (40.0, SHARP),      # 1.43 deg/m
    ])
    def test_sharpness_classes(self, radius, expected):
        points, s, yaw, L = stadium(radius=radius)
        turns = segment_turns(points, s, yaw, L)
        assert len(turns) == 2
        assert all(t.sharpness == expected for t in turns)
        assert all(t.direction == "right" for t in turns)

    def test_left_direction(self):
        points, s, yaw, L = stadium(radius=100.0, direction="left")
        turns = segment_turns(points, s, yaw, L)
        assert len(turns) == 2
        assert all(t.direction == "left" for t in turns)

    def test_turn_extent_roughly_matches_arc(self):
        points, s, yaw, L = stadium(radius=100.0)
        turns = segment_turns(points, s, yaw, L)
        arc = np.pi * 100.0  # semicircle length
        for t in turns:
            # Smoothing blurs boundaries by ~half the window on each side.
            assert turn_length(t, L) == pytest.approx(arc, abs=15.0)


class TestSplitAndMerge:
    def test_split_on_sharpness_change(self):
        # Moderate 90-degree arc flowing directly into a sharp 90-degree arc.
        points, s, yaw, L = build_track([
            ("straight", 200.0),
            ("arc", 100.0, 90.0, "right"),
            ("arc", 40.0, 90.0, "right"),
            ("straight", 200.0),
            ("arc", 100.0, 180.0, "right"),
        ])
        turns = segment_turns(points, s, yaw, L)
        # Expect: moderate + sharp continuation (split), then the closing turn.
        assert len(turns) == 3
        first, second = turns[0], turns[1]
        assert (first.sharpness, second.sharpness) == (MODERATE, SHARP)
        assert not first.is_continuation
        assert second.is_continuation
        assert second.s_start == pytest.approx(first.s_end)
        # Continuation keeps only its s_start marker: continuous beep re-triggers.
        assert second.marker_s == [second.s_start]
        assert second.approach_beep_nums == []

    def test_short_gap_merges_same_direction_turns(self):
        points, s, yaw, L = build_track([
            ("straight", 200.0),
            ("arc", 100.0, 45.0, "right"),
            ("straight", 3.0),  # < MERGE_GAP_M
            ("arc", 100.0, 45.0, "right"),
            ("straight", 200.0),
            ("arc", 100.0, 270.0, "right"),
        ])
        turns = segment_turns(points, s, yaw, L, smoothing_window_m=1.0)
        assert len(turns) == 2  # front pair merged into one, plus the closer
        front = turns[0]
        expected = 2 * (100.0 * np.radians(45.0)) + 3.0
        assert turn_length(front, L) == pytest.approx(expected, abs=6.0)

    def test_short_kink_dropped(self):
        # A 2.8 m sharp kink is below MIN_TURN_LENGTH_M and must vanish.
        points, s, yaw, L = build_track([
            ("straight", 200.0),
            ("arc", 40.0, 4.0, "right"),
            ("straight", 200.0),
        ])
        turns = segment_turns(points, s, yaw, L, smoothing_window_m=1.0)
        assert turns == []


class TestSeam:
    def test_turn_spanning_array_start(self):
        points, s, yaw, L = stadium(radius=100.0)
        # Roll the arrays so a semicircle spans index 0 (mid-arc at the seam).
        k = 200 + 157  # mid of first arc
        points2 = np.roll(points, -k, axis=0)
        yaw2 = np.roll(yaw, -k)
        seg_len = np.hypot(*np.diff(points2, axis=0).T)
        s2 = np.concatenate([[0.0], np.cumsum(seg_len)])
        L2 = float(s2[-1] + np.hypot(*(points2[0] - points2[-1])))

        turns = segment_turns(points2, s2, yaw2, L2)
        assert len(turns) == 2
        total = sum(turn_length(t, L2) for t in turns)
        assert total == pytest.approx(2 * np.pi * 100.0, abs=30.0)


class TestMarkers:
    def test_marker_placement_and_numbering(self):
        points, s, yaw, L = stadium(radius=100.0)
        turns = segment_turns(points, s, yaw, L)
        t = turns[0]
        assert t.approach_beep_nums == [1, 2, 3]
        assert len(t.marker_s) == 4
        assert t.marker_s[-1] == pytest.approx(t.s_start)
        d = TUS_MARKER_SPACING_M
        for i, m in enumerate(t.marker_s[:-1]):
            expected = (t.s_start - (3 - i) * d) % L
            assert m == pytest.approx(expected, abs=1e-6)

    def test_markers_inside_previous_turn_are_clamped(self):
        # 30 m gap between two sharp turns: markers at -60 and -40 land inside
        # the previous turn and must be dropped; -20 (in the gap) survives.
        points, s, yaw, L = build_track([
            ("straight", 150.0),
            ("arc", 40.0, 90.0, "right"),
            ("straight", 30.0),
            ("arc", 40.0, 90.0, "right"),
            ("straight", 150.0),
            ("arc", 80.0, 180.0, "right"),
        ])
        turns = segment_turns(points, s, yaw, L, smoothing_window_m=1.0)
        assert len(turns) == 3
        second = turns[1]
        assert not second.is_continuation
        assert second.approach_beep_nums == [3]
        assert len(second.marker_s) == 2  # surviving approach marker + s_start
