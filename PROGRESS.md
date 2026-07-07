# Phase 2 Attentional Stack — Implementation Progress

Full approved plan: `~/.claude/plans/partitioned-cuddling-piglet.md` (design also summarized below).
Branch: `bo-module`. Session date: 2026-07-05.

## Decisions locked (user-approved)
- Scope: attentional stack = TUS + turn detection + `attentional_shift_strength` + 6-D YAML. Feedback module DEFERRED.
- `steer_max` INCLUDED in BO search space.
- TUS spec per RAD paper (CHI 2018 preprint, recovered via Wayback): 3 discrete beeps at markers 20 m apart + continuous 4th beep spanning the turn; direction = stereo side; sharpness = pitch (soft <0.3°/m, moderate, sharp >1°/m); turn-number TTS deferred.
- Intention preservation: attentional gain shift must NEVER start before the turn's first beep.

## Status

- [x] `game/turns.py` — centerline builder (walks `map.get_waypoint().next()` from fixed spawn; the world's `generate_waypoints` cloud is UNORDERED and unusable), pure `segment_turns`, `TrackProfile` (event table + own KD-tree), `TusTracker` (interval-membership event firing; handles wrap/reverse/teleport/off-track).
- [x] `tests/test_turns.py` + `tests/test_tus_tracker.py` — 19/19 passing (`.venv/bin/python -m pytest tests/ -q`). Also `tests/conftest.py` + `tests/trackgen.py` (synthetic track builder). Fixes found by tests: sub-run merge threshold must be ≥ smoothing window (transition-zone artifacts); segments sorted by s_start + renumbered (rotation-independent ids); seam-gap samples forced straight.
- [x] `game/audio.py` rewrite — render_block refactor, beep + continuous synth (freqs 440/660/880 Hz, raised-cosine envelopes, phase-continuous), attentional gain ramp (MAX_ATTENUATION=0.5, ramp 0.4 s), `update_tus()`, `open_stream=False` for tests. NOT yet covered by tests (next step).
- [x] `tests/test_audio_render.py` (12 tests: pitch/pan/duration, continuous span + retrigger click check, gain ramp, intention-preservation regression, legacy equivalence) + `scripts/render_tus_demo.py` → `results/tus_demo.wav` rendered (18 s scripted lap, listen to verify). 31/31 tests passing.
- [x] Plumbing: `app.py` (TrackProfile+TusTracker init after spawn, startup turn-table print, update_tus in 50 ms audio block, new kwargs incl. input_device), `main.py` (--attentional-shift-strength, --no-tus, --tus-marker-spacing, forwards --input)
  - both bug fixes applied: `controller.py` `round(steer,4)`; `app.py` builtin-`input` → `input_device`; `_switch_vehicle` self.args/self.ctrl_kwargs → real attrs
- [x] `config/phase2.yml` written (6 params) + `optimize.py`: plot_bo_step guarded behind `train_X.shape[-1] == 1`; CONFIG_FILE=config/phase2.yml, VERSION=v2. Verified kwarg flow: YAML names → tensor_to_kwargs → RacingApp(**kwargs), all 6 match.
- [x] VERIFIED (2026-07-06): py_compile clean on all edited files; phase2.yml extracts 6 params with (2, 6) bounds and round-trips to RacingApp kwargs correctly; full pytest suite 31/31 passing.

## GCP infra (2026-07-06)
Original `carla-server` (asia-east2-c, T4) hit `ZONE_RESOURCE_POOL_EXHAUSTED` on restart — global T4 shortage that day (confirmed: every US zone + most EU/Asia zones failed identical capacity error in a 40+ zone fan-out). Recreated from an image of the original boot disk (no reinstall) as **`carla-europecentral2b`** in zone `europe-central2-b`, project `amlichomework`. Image `carla-server-image` kept around in case this zone dies too (delete if unwanted — ~$1-3/mo storage).
- External IP: `34.118.63.22` (ephemeral — will change if the VM restarts; re-check with `gcloud compute instances describe carla-europecentral2b --project=amlichomework --zone=europe-central2-b --format='value(networkInterfaces[0].accessConfigs[0].natIP)'`)
- Firewall `allow-carla` (tcp:2000-2001, 0.0.0.0/0) is network-wide, no per-instance config needed.
- CARLA install found at `/home/krish/CARLA_0.9.16` (user `krish`, not `krishrveera`); launched headless via `sudo -u krish tmux new-session -d -s carla './CarlaUE4.sh -RenderOffScreen -quality-level=Low'` — **currently RUNNING**, confirmed listening on 2000/2001, GPU active. Survives SSH disconnect (tmux); will NOT survive a VM restart — relaunch the same tmux command if it reboots.
- `nvidia-smi` came up clean on the new VM with no fixes needed (driver 535.309.01, CUDA 12.2) — image transplant preserved the GPU driver correctly.
- Client side (this Mac): `.venv` only has test deps (numpy/scipy/yaml/pytest/sounddevice/torch) — still needs `pip install -r requirements.txt` for the full `carla` wheel + `pygame` before `python main.py --host 34.118.63.22 --port 2000` will work.

## ALL AUTOMATED WORK COMPLETE — remaining steps need a human + CARLA server
- [ ] MANUAL (needs CARLA server + human driver, cannot be automated):
  - Empirical check: centerline loop closure on Town04 from spawn (84.87, 370.80, 18.00); verify `RIGHT_TURN_SIGN=+1` (UE yaw clockwise-positive — analytic, unverified empirically)
  - Smoke test: drive lap, per turn expect 3 beeps → continuous, correct side/pitch; `--attentional-shift-strength 1.0` ducks slider only AFTER first beep; `--no-tus` == old behavior
  - Marker spacing pilot: 20 m at 8 m/s = 7.5 s notice (paper: 1.7 s); speed-scaled candidate ≈ 4.5 m via `--tus-marker-spacing`

## Key YAML ranges (calibrated to actual main.py defaults; plan-doc ranges were 10–200× off)
```yaml
warp_factor:                default 1.0,   range "(0.0, 6]"
steer_increment:            default 0.001, range "[0.0002, 0.005]"
steer_max:                  default 0.1,   range "[0.02, 0.5]"
steer_decay:                default 0.2,   range "[0.05, 0.95]"
brake_increment:            default 0.3,   range "[0.06, 1.0]"
attentional_shift_strength: default 0.0,   range "[0.0, 1.0]"
```

## Open questions for team (documented, not blocking)
- `attentional_shift_onset`: likely intention-violating (early-warning cue); discuss with Antti/Brian.
- MAX_ATTENUATION 0.5 needs pilot testing.
- 6-D BO with ~16 human laps is exploration-starved: bump `n_init_samples`, or stage the optimization (warp first, then steering).
