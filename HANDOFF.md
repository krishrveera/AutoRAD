# AutoRAD Phase 2 — Handoff (2026-07-06)

Read this first if you're picking up on a new machine with no chat history.
Companion file: `PROGRESS.md` (granular build checklist, what's tested, what
isn't). This file is the "why" and "what's next"; `PROGRESS.md` is the "what
got built, line by line."

---

## 1. What this project is

AutoRAD adapts the **Racing Auditory Display (RAD)** — Smith & Nayar, CHI 2018
— to individual blind/low-vision players via Bayesian Optimization. RAD lets
a blind player drive a car in the CARLA simulator using two audio channels:

- **Sound slider**: continuous engine tone, panned left/right to encode how
  close the car is to hitting either edge of the track.
- **Turn Indicator System (TUS)**: 4 beeps that fire at distance markers
  before a turn, encoding direction (stereo side), sharpness (pitch), and
  turn length (a continuous 4th beep spans the turn).

The BO loop watches a human drive laps under different parameter settings and
searches for the settings that produce the fastest/most comfortable driving,
without a human having to manually tune anything.

## 2. What existed before this session (Phase 1, by Nghi)

One optimizable knob: `warp_factor` (a pan-curve exponent). The evaluation
loop in `optimize.py`: BO picks a `warp_factor` → you drive one lap → wall
clock time is recorded → `score = -log(time)` → BO updates its model → picks
the next value → repeat 10 times. Results land in
`results/optimization/v1/`: `optimization_history.csv`,
`optimization_results.json`, and one `bo_step_Iteration_N.png` per
iteration — a 2-line plot (GP surrogate + acquisition function) because with
1 knob, the search space is just a line on an x-axis.

**Nghi's v1 result** (already in the repo, for reference):
default `warp_factor=1.0` → 195.6s lap. Best found (`warp_factor≈1.17`) →
73.4s. **62.5% faster.** That's the existence proof this whole approach
works.

## 3. What Phase 2 (this session) adds, and why

Two things stacked on the same skeleton:

1. **More knobs.** Search space goes from 1 dimension to 6:
   `warp_factor`, `steer_increment`, `steer_max`, `steer_decay`,
   `brake_increment`, `attentional_shift_strength`.
2. **A genuinely new audio feature.** The RAD paper describes *two* channels
   (sound slider + turn indicator beeps). This codebase only ever had the
   slider — the beep system (TUS) did not exist. I built it
   (`game/turns.py` + rewritten `game/audio.py`). `attentional_shift_strength`
   is a knob that controls how much the game shifts emphasis between the two
   channels depending on context (straightaway vs. turn) — it only makes
   sense once TUS exists.

### The hypothesis

"Giving BO more, and better, things to tune produces a setup that's
faster/more comfortable than tuning `warp_factor` alone did." Concretely,
two things we can check with this build:

- Does the beep system + attentional balance help players react to turns
  *without* handing them extra unfair information (i.e. it must not become
  an early-warning cue — this is the "intention preservation" rule the whole
  RAD project is built on)?
- Does searching 6 numbers instead of 1 still find something clearly better
  than default, given that each data point costs one real human-driven lap?
  (Flagged risk: 6-D needs far more samples than 1-D to search properly, and
  we probably won't get as many laps as the math wants — a staged approach,
  optimizing `warp_factor` first then freezing it and optimizing steering,
  is the fallback if joint 6-D turns out too sample-starved.)

**Not yet being tested**: whether the setup *feels* more comfortable
(subjective player feedback). That needs a feedback-survey module which was
explicitly deferred this round — still lap-time-only, same objective
function shape as v1.

### How the test actually runs

Identical mechanism to v1, just wider: `python optimize.py` drives the same
4-phase loop (initial samples → default baseline → BO iterations → save
results), same `evaluate_my_program` (one human lap → `-log(time)` score),
same output files. The comparison that matters: default-row score vs.
best-found-row score in `optimization_history.csv`, same shape as the 62.5%
number above, just across 6 knobs instead of 1.

### What got reused vs. rewritten

**Reused untouched**: `opt/params.py`, `opt/extract_params.py` — the
YAML→BoTorch glue was already written to handle any number of parameters,
not just one. Expanding to 6 knobs needed zero changes there. `optimize.py`'s
core loop (fit GP → pick next point → ask human to drive → record score) is
unchanged.

**New/changed**: `config/phase2.yml` (6 params vs. minimal.yml's 1),
`game/audio.py` (rewritten: TUS channel + attentional gain mixing),
`game/turns.py` (new: figures out where turns are on the track from CARLA
waypoints, and fires beep events as the car drives), `main.py`/`game/app.py`
(new CLI flags), `game/controller.py` (one-line bug fix — steering was
being rounded to 3 possible values, `-0.1/0/+0.1`, making the steering knobs
meaningless to optimize; fixed to a finer rounding).

### ⚠️ Known gap: the `bo_step_Iteration_N.png` plots won't be produced as-is

Those plots are inherently 1-dimensional (x-axis = the one knob). With 6
knobs there's no single x-axis to draw. I guarded the plotting code
(`optimize.py`) to skip instead of crash when the search space has more than
1 dimension — meaning **running `optimize.py` against `config/phase2.yml`
right now will NOT produce those PNGs.** You still get the full
`optimization_history.csv` / `optimization_results.json` (every run's 6
parameter values + score), just not that per-iteration picture. If a visual
is wanted for the 6-D case, that's new code (e.g. one small plot per
dimension, or a parallel-coordinates plot) — not yet written, needs a design
decision on what to show.

---

## 4. Current repo state

**Nothing is committed.** All Phase 2 work is sitting as uncommitted changes
in the working tree on the Mac at `/Users/krishrveera/Desktop/AutoRAD`
(branch `bo-module`). If you're moving to the Windows PC, you need to bring
the actual working tree over (zip it, git-bundle it, whatever) — a plain
`git clone` of the remote will NOT have this work.

Changed/new files: `game/app.py`, `game/audio.py`, `game/controller.py`,
`main.py`, `optimize.py`, `game/turns.py` (new), `config/phase2.yml` (new),
`tests/` (new, 31 passing unit tests, no CARLA needed to run them),
`scripts/render_tus_demo.py` (new), `results/tus_demo.wav` (new — a
synthetic audio render you can listen to without CARLA), `PROGRESS.md`,
this file.

All 31 automated tests pass (`pytest tests/ -q`). Nothing requiring a live
CARLA server has been exercised yet — that part needs a human driving,
which needs the Windows PC.

## 5. The GCP CARLA server (already running, as of this session)

Original VM `carla-server` (asia-east2-c, project `amlichomework`) hit
`ZONE_RESOURCE_POOL_EXHAUSTED` on restart — a global T4 GPU shortage that
day (confirmed across 40+ zones). Recreated from an image of the original
boot disk (so nothing had to be reinstalled) as:

- **Instance name**: `carla-europecentral2b`
- **Zone**: `europe-central2-b`, project `amlichomework`
- **External IP**: `34.118.63.22` (ephemeral — check again if the VM has
  been restarted since:
  `gcloud compute instances describe carla-europecentral2b --project=amlichomework --zone=europe-central2-b --format='value(networkInterfaces[0].accessConfigs[0].natIP)'`)
- **CARLA install path on the VM**: `/home/krish/CARLA_0.9.16` (note: user
  is `krish`, not `krishrveera`)
- **Status as of this session**: CARLA launched headless in a `tmux` session
  named `carla`, confirmed listening on ports 2000/2001, GPU actively
  loaded. Survives SSH disconnects. **Will NOT survive a VM reboot** — if it
  ever stops responding, SSH in and run:
  ```bash
  gcloud compute ssh carla-europecentral2b --project=amlichomework --zone=europe-central2-b \
    --command="sudo -u krish tmux new-session -d -s carla 'cd /home/krish/CARLA_0.9.16 && ./CarlaUE4.sh -RenderOffScreen -quality-level=Low; exec bash'"
  ```
- Firewall rule `allow-carla` (tcp:2000-2001, open to `0.0.0.0/0`) is
  network-wide, no per-instance config needed regardless of which zone the
  server ends up in.
- A reusable disk image, `carla-server-image`, was kept in the project in
  case this zone also runs out of capacity later (small ongoing storage
  cost, ~$1-3/month — delete if you don't want it hanging around).
- Original `carla-server` (asia-east2-c) is still there, `TERMINATED`, not
  costing compute. Left alone, not deleted.
- GCP account in use: `krishvusf@gmail.com` (switched from
  `krv2123@columbia.edu` mid-session because the Columbia account didn't
  have access to project `amlichomework`).

## 6. Why this can't run on the Mac, and what to do on Windows

The `carla` pip package (the Python connector to the CARLA simulator) has
**never published a macOS wheel**, for any version — checked every release
0.9.3 through 0.9.16, all of them are Linux (`manylinux`) and Windows
(`win_amd64`) only. Not a version-pinning problem, not fixable by
reinstalling — the package itself was never built for Mac. So `main.py` (the
actual playable game) can only run on Windows or Linux.

**On the Windows PC:**

```bash
# 1. Bring the repo over (whole working tree, since nothing is committed yet)

# 2. Python 3.11 or 3.12, then:
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify
python -c "import carla, torch, botorch, sounddevice; print('OK')"

# 4. Run against the cloud CARLA server
python main.py --host 34.118.63.22 --port 2000 --input keyboard
```

If that IP has changed (VM restarted), re-check it with the `gcloud`
command in section 5.

**Useful flags for trying out the new Phase 2 features individually**
(see `PROGRESS.md` for the full manual test checklist):

```bash
python main.py --host 34.118.63.22 --no-tus                          # old Phase 1 sound, A/B reference
python main.py --host 34.118.63.22 --attentional-shift-strength 1.0  # max channel-ducking in turns
python main.py --host 34.118.63.22 --steer-max 0.5 --steer-increment 0.005  # twitchy steering (range ceiling)
```

**Running the actual BO evaluation** (once you're happy the knobs behave):

```bash
python optimize.py   # currently points at config/phase2.yml, version "v2"
```

Watch `results/optimization/v2/optimization_history.csv` for the real
signal: best score vs. default-baseline score, same comparison Nghi's
62.5% number came from.

---

## 7. Open questions / things not yet decided

- No visualization built yet for the 6-D BO result (see section 3's warning).
- `attentional_shift_onset` (a second attentional parameter, controlling how
  early the emphasis shift begins) was flagged in the original plan doc as
  likely violating intention preservation — deliberately not built, listed
  as an open question for the team.
- `MAX_ATTENUATION` (how strongly the de-emphasized channel gets quieted,
  currently 0.5) hasn't been pilot-tested with a real driver.
- `TUS_MARKER_SPACING_M` (currently 20m, matching the paper) assumes ~35m/s
  driving; this project's speed limit is 8 m/s, so the warning comes ~4x
  earlier than the paper intended (7.5s vs. 1.7s notice) — worth trying
  `--tus-marker-spacing 4.5` and seeing which feels right.
- 6-D BO with a realistic number of human-driven laps is likely
  sample-starved (flagged risk); a staged approach (optimize `warp_factor`
  first, freeze it, then optimize the rest) is the fallback if joint search
  doesn't converge to anything meaningful.
- `steer_max` was deliberately included in the search space (a team decision
  made mid-session) despite being borderline on the "does this change what
  the game demands of the player" test that the whole knob taxonomy is built
  around — worth a second look once there's driving data.
