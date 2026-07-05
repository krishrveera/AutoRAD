# AutoRAD

**Auditory driving assistance in CARLA, auto-tuned with Bayesian Optimization.**

AutoRAD turns the road around a car into *sound*. As you drive a vehicle in the
[CARLA](https://carla.org/) simulator, the engine of the car is panned left and
right in your headphones to tell you which side of the lane has more room — a
cue we call the **RAD ratio** (Ratio of Available Distance). The project then
uses [BoTorch](https://botorch.org/) Bayesian Optimization to automatically
find the audio-warping parameter that lets a human driver complete a lap as
fast as possible.

For the *what / how / why* of the system, see **[PROJECT.md](PROJECT.md)**.
This file is just about getting it running.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **OS** | Windows 10/11 or Linux (Ubuntu 20.04+). CARLA does not ship for macOS. |
| **Python** | **3.11 or 3.12** — this is the overlap between CARLA 0.9.16 (3.10–3.12) and BoTorch (3.11+). |
| **GPU** | A dedicated GPU (NVIDIA recommended, ~6 GB+ VRAM) for CARLA's renderer. |
| **CARLA Simulator** | **0.9.16**, the standalone server package (separate from the `carla` Python wheel). |
| **Audio** | Headphones strongly recommended — the whole point is stereo panning. |
| **Gamepad** | Optional. An Xbox-style controller gives smoother input than the keyboard. |

---

## 2. Install the CARLA simulator server

The `carla` pip package is only the *client library*. You also need the actual
simulator engine running as a server.

1. Download the **CARLA 0.9.16** package from the
   [official release page](https://carla.org/2025/09/16/release-0.9.16/) or the
   [GitHub releases](https://github.com/carla-simulator/carla/releases).
2. Extract it somewhere permanent, e.g. `C:\CARLA_0.9.16` or `~/CARLA_0.9.16`.
3. Launch the server (leave this running in its own terminal):

   ```bash
   # Windows
   CarlaUE4.exe

   # Linux
   ./CarlaUE4.sh
   ```

   By default it listens on `localhost:2000`. To run it headless / lower-spec:

   ```bash
   ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
   ```

> Docs: [CARLA quick-start](https://carla.readthedocs.io/en/latest/start_quickstart/).

---

## 3. Set up the Python environment

Use a fresh virtual environment so the heavy ML dependencies stay isolated.

```bash
# from the repo root
python3.12 -m venv .venv

# activate it
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows (PowerShell)

# upgrade pip, then install everything
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs two groups of dependencies:

- **Core simulator/game** — `carla`, `numpy`, `pygame`, `sounddevice`, `scipy`,
  `pyyaml`.
- **Optimization & analysis** — `torch`, `botorch`, `gpytorch`, `pandas`,
  `matplotlib`, `seaborn`. (Only needed if you run `optimize.py` or the analysis
  scripts; the game itself runs without them.)

> Docs: [BoTorch getting started](https://botorch.org/docs/getting_started/),
> [`carla` on PyPI](https://pypi.org/project/carla/).

### Verify the install

With the CARLA server running:

```bash
python -c "import carla, torch, botorch, sounddevice; print('OK')"
python scripts/config.py --inspect       # prints map, version, actor counts
```

---

## 4. (Optional) Load a custom racetrack

The `maps/` folder holds OpenDRIVE (`.xodr`) racetracks at various lane widths.
By default the app reuses whatever world the server already has loaded. To load
one of the bundled tracks into the running server:

```bash
python scripts/config.py --xodr-path maps/racetrack_20.0m.xodr
```

Helper tools for working with tracks:

- `scripts/draw_map.py` — draws a pillar at every waypoint to confirm the track
  parsed as one continuous loop.
- `scripts/change_width.py <file.xodr> <width>` — rewrites driving-lane widths.
- `scripts/get_spawn_point.py` — captures the spectator camera's transform so
  you can set a fixed spawn/pole position.

---

## 5. Run the simulator

With the CARLA server running and the venv active:

```bash
# Gamepad (default)
python main.py

# Keyboard
python main.py --input keyboard
```

**Controls**

| Action | Keyboard | Gamepad (Xbox layout) |
|--------|----------|-----------------------|
| Throttle | ↑ | Left stick up |
| Brake / reverse | ↓ | Left stick down |
| Steer | ← / → | Left stick X |
| Disable autopilot | Enter | A |
| Quit | Close window | — |

You drive a lap; when you cross the finish line a completion message prints and
the app shuts down cleanly. Useful flags (see `python main.py --help` for all):

```bash
python main.py --warp-factor 2.0 --speed-limit 8.0 --time-horizon 120
```

---

## 6. Run the Bayesian Optimization

This is the research workflow: it repeatedly asks **you** to drive a lap with
different `warp_factor` values and learns which one minimizes your lap time.

1. Start the CARLA server.
2. Edit the search space in `config/minimal.yml` if desired.
3. Run:

   ```bash
   python optimize.py
   ```

4. Follow the prompts — press **ENTER** when you're ready to drive each lap.

Results land in `results/optimization/<version>/`:

- `optimization_results.json` — best parameters and improvement vs. default.
- `optimization_history.csv` — every run's parameters, score, and lap time.
- `bo_step_Iteration_*.png` — the surrogate model + acquisition function per step.

### Supporting experiments

```bash
# Quantify run-to-run human-driving noise across warp factors
python scripts/approximate_noise.py
python scripts/analyze_noise.py        # plots the collected CSV

# Re-create the BO decision plots offline from logged history
python scripts/replay_opt.py

# Compare KD-tree RAD vs. native CARLA-API RAD (speed + accuracy)
python scripts/benchmark.py
```

---

## 7. Project layout

```
main.py                 CLI entry point — launches the playable game
optimize.py             Human-in-the-loop Bayesian Optimization driver
requirements.txt        Pinned dependencies
config/minimal.yml      Parameter search space for the optimizer

game/                   The real-time application
  app.py                RacingApp — wires subsystems together, owns the loop
  world.py              CarlaWorld — server connection, spawning, waypoints
  ego.py                Ego — vehicle state + RAD-ratio computation
  audio.py              AudioManager — stereo engine-sound synthesis
  render.py             DisplayManager / CameraSensor — PyGame window + HUD
  controller.py         Keyboard & gamepad vehicle control

opt/                    Optimizer plumbing
  params.py             Parameter / ParameterSpace (BoTorch <-> kwargs glue)
  extract_params.py     YAML config -> ParameterSpace

scripts/                Standalone tools & experiments (see sections 4 & 6)
maps/                   OpenDRIVE (.xodr) racetracks
assets/audio/           Engine sound samples
results/                Generated plots, CSVs, JSON metrics
```

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `RuntimeError: time-out ... waiting for the simulator` | CARLA server isn't running, or wrong `--host`/`--port`. |
| `ModuleNotFoundError: carla` | Wrong Python version — must be 3.10–3.12; reinstall the wheel in the venv. |
| `ImportError: botorch` | Optimization deps not installed; rerun `pip install -r requirements.txt` on Python 3.11/3.12. |
| No sound / wrong device | Check OS output device; `sounddevice` uses the system default. Use headphones. |
| Gamepad ignored | App falls back to keyboard if no controller is detected — plug in before launching. |
| Car spawns inside geometry | The spawn logic retries with a Z-offset; if it persists, pick a different map/spawn. |
