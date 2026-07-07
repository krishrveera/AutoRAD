"""Render a synthetic TUS lap to a WAV file for listening inspection.

No CARLA or audio device needed. Simulates: straightaway -> three approach
beeps (2.5 s apart, as at 8 m/s with 20 m markers) -> a sharp right turn with
the continuous fourth beep -> straightaway, with the attentional gain shift
audible when --shift > 0.

Usage
-----
    python scripts/render_tus_demo.py [--shift 1.0] [--out results/tus_demo.wav]
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.audio import AudioManager
from game.turns import BeepEvent, ContState, TusUpdate

FS = 44100
BLOCK = 512


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift", type=float, default=1.0,
                        help="attentional_shift_strength to render with")
    parser.add_argument("--out", default="results/tus_demo.wav")
    args = parser.parse_args()

    mgr = AudioManager(sample_rate=FS, warp_factor=2.0,
                       attentional_shift_strength=args.shift, open_stream=False)
    mgr.update_state(speed=8.0, ratio=0.5)

    def at(seconds):
        return int(seconds * FS / BLOCK)

    direction, sharpness = "right", 2
    timeline = {
        at(3.0): TusUpdate([BeepEvent(1, direction, sharpness, 0)], None, True),
        at(5.5): TusUpdate([BeepEvent(2, direction, sharpness, 0)], None, True),
        at(8.0): TusUpdate([BeepEvent(3, direction, sharpness, 0)], None, True),
        at(10.5): TusUpdate([], ContState(direction, sharpness, 0), True),
        at(14.5): TusUpdate([], None, False),
    }

    total_blocks = at(18.0)
    chunks = []
    for i in range(total_blocks):
        if i in timeline:
            mgr.update_tus(timeline[i])
        chunks.append(mgr.render_block(BLOCK))
    signal = np.concatenate(chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(pcm.tobytes())
    print(f"Wrote {out_path} ({len(signal) / FS:.1f} s, shift={args.shift})")
    print("Timeline: beeps at 3.0/5.5/8.0 s, continuous turn 10.5-14.5 s.")


if __name__ == "__main__":
    main()
