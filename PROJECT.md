# AutoRAD — Project Deep Dive

> What the project is, how it works, and **why** it works the way it does.
> For installation and commands, see [README.md](README.md).

---

## 1. The idea in one paragraph

A sighted driver knows where the edges of the road are by *looking*. AutoRAD
asks: **can a driver stay on the road using only their ears?** It continuously
estimates how much drivable room exists to the left versus the right of the
car, collapses that into a single "where am I in the lane" number, and renders
that number as **stereo panning of the engine sound**. Hug the left wall and the
engine roars in your right ear telling you to come back; sit dead-center and the
sound is balanced. On top of this perception layer, AutoRAD runs **Bayesian
Optimization** to discover the audio-mapping setting that lets a real human lap
the track fastest — i.e. it tunes the *interface* to the *human*, treating the
human-in-the-loop as a noisy black-box objective.

This sits at the intersection of three fields: **autonomous-driving simulation**
(CARLA), **sonification / accessible interfaces** (the RAD audio cue), and
**black-box optimization** (BoTorch). The name is *Auto*nomous + *RAD* (Ratio of
Available Distance).

---

## 2. System architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              CARLA Server (UE4)              │
                        │   physics · rendering · world state · maps   │
                        └───────────────▲───────────────┬─────────────┘
                                        │ control        │ sensor + state
                                        │                ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                          RacingApp  (game/app.py)                        │
   │                       owns the synchronous game loop                     │
   │                                                                          │
   │   ┌───────────┐   ┌──────────┐   ┌──────────────┐   ┌────────────────┐   │
   │   │ Controller │  │ CarlaWorld│  │     Ego      │   │  DisplayManager │   │
   │   │ kbd/gamepad│  │ spawn +   │  │ speed + RAD  │   │ camera + HUD    │   │
   │   │  → control │  │ waypoints │  │ + finish line│   │                 │   │
   │   └───────────┘   └────┬─────┘   └──────┬───────┘   └────────────────┘   │
   │                        │ KD-tree         │ RAD ratio                      │
   │                        ▼                 ▼                                │
   │                  ┌──────────────────────────────┐                        │
   │                  │   AudioManager (game/audio)   │  → stereo headphones   │
   │                  │ sawtooth engine + EP panning  │                        │
   │                  └──────────────────────────────┘                        │
   └────────────────────────────────────────────────────────────────────────┘
                                        ▲
                                        │ lap time (objective)
   ┌────────────────────────────────────┴───────────────────────────────────┐
   │                        optimize.py  (BoTorch loop)                       │
   │   GP surrogate · qNoisyExpectedImprovement · suggests next warp_factor   │
   │            opt/params.py · opt/extract_params.py · config/*.yml          │
   └─────────────────────────────────────────────────────────────────────────┘
```

The clean separation matters: the **game** (`game/`) knows nothing about
optimization, and the **optimizer** (`optimize.py`, `opt/`) treats the game as
an opaque function `params → lap_time`. That boundary is what lets the same code
be both a playable demo and a research apparatus.

---

## 3. How the simulation works

### 3.1 Deterministic, synchronous stepping

`CarlaWorld` (`game/world.py`) puts the server into **synchronous mode** with a
fixed timestep (`delta`). In this mode the simulator does not advance until the
client calls `tick()`. **Why:** reproducibility. Audio timing, trajectory
prediction, and lap-time measurement all depend on the physics advancing in
lock-step with our code rather than at the mercy of a variable frame rate.

### 3.2 The game loop

`RacingApp.run()` (`game/app.py`) is a textbook fixed loop:

1. `world.tick()` — advance physics one step.
2. `controller.process_control()` + `_handle_events()` — apply player input.
3. **Throttled** audio update (every `update_gap` ms): recompute Ego state and
   push speed + RAD ratio into the audio engine.
4. `display.render()` — blit the chase-camera frame and HUD.
5. Check the finish line; end the lap if crossed.

**Why throttle the audio update** (step 3 runs slower than the physics): the RAD
computation is the most expensive thing per frame, and human hearing doesn't
need kilohertz-rate pan updates. Decoupling the perception rate from the physics
rate keeps the loop smooth.

### 3.3 Input handling

`controller.py` defines a `ControlObject` base with two subclasses
(`KeyboardControlObject`, `GamepadControlObject`) chosen by `make_controller()`.
Both share one throttle model: a **burst increment** from rest up to a threshold,
then a slower **normal increment**, with an easing band that bleeds off throttle
as the car approaches `speed_limit`. **Why the shared ramp:** so that every
throttle CLI parameter affects keyboard and gamepad identically — the only
difference is the *target* (a fixed ceiling for keyboard vs. proportional to
stick deflection for gamepad). This keeps experiments comparable across input
devices.

---

## 4. The core algorithm: the RAD ratio

This is the heart of the project, in `game/ego.py`.

### 4.1 What we want

A single number `ratio ∈ [0, 1]`:

- `0.0` → a wall is very close on the **left** (lots of room to the right),
- `1.0` → a wall is very close on the **right**,
- `0.5` → centered / balanced.

### 4.2 How we compute it — kinematic "what-if" trajectories

For each frame we ask two counterfactual questions: *"If I cranked the wheel
hard left right now, how far would I travel before leaving the road? And hard
right?"* Those two arc-lengths, `dist_left` and `dist_right`, define the ratio.

`_predict_trajectory_to_impact_kdtree()` rolls out a **bicycle kinematic model**
from the car's current pose:

```
x   += v·cos(yaw)·dt
y   += v·sin(yaw)·dt
yaw += (v / wheelbase)·tan(steer_angle)·dt      # until a 90° turn accumulates
```

- `steer_angle` = the wheel's max steer angle × `steer_intensity`.
- The turn is capped at 90° of accumulated yaw so the projection is a realistic
  evasive arc, not an endless circle.
- Integration runs for `time_horizon` seconds at step `dt`.

### 4.3 Why a KD-tree instead of CARLA's API

The naïve way to test "is this point still on the road?" is
`carla_map.get_waypoint(point, project_to_road=False)` per trajectory point —
but that's a relatively slow RPC-style call, and we make *hundreds* of them per
frame (two directions × `time_horizon/dt` steps).

Instead, at startup `CarlaWorld._fit_waypoints_map()` samples every driving lane
into a NumPy array of `[x, y, lane_width]` and builds a **`scipy.spatial.cKDTree`**
over the `(x, y)` columns. A trajectory point is "off-road" when its distance to
the nearest waypoint exceeds **half that waypoint's lane width**. Points are
checked in **vectorized chunks** (`_query_chunk_for_impact`), and the rollout
stops at the first chunk containing an off-road point.

**Why this is correct *and* fast:** `scripts/benchmark.py` exists precisely to
prove it. It runs both methods side-by-side every frame and reports (a) the
speedup and (b) the mean-squared deviation between the two RAD values — i.e. the
KD-tree method is many times faster while staying numerically faithful to the
ground-truth API.

### 4.4 Turning two distances into one ratio

`_calculate_rad()` handles the edge cases explicitly:

| Case | Meaning | Ratio |
|------|---------|-------|
| both `None` | neither arc hit a boundary within the horizon | `0.5` (safe/centered) |
| left `None` | only the right side has a near boundary | `dist_right / (dist_right + maxlen + 1)` |
| right `None` | only the left side has a near boundary | `dist_left / (dist_left + maxlen + 1)` |
| either `-1` | car is essentially stationary | skip (don't update) |
| both finite | normal case | `dist_left / (dist_left + dist_right)` |

**Why normalize against `maxlen` in the one-sided cases:** if only one side ever
hits a wall, there's no opposing distance to divide by, so we fold in the
maximum possible trajectory length as a stable denominator. This keeps the
number continuous and bounded as boundaries appear and disappear, avoiding jarring
audio jumps.

The ratio is only updated above `min_speed` — RAD on a stationary car is noise.

### 4.5 Finish-line detection

`_check_finish_line()` doesn't trust CARLA's lane topology (custom `.xodr`
tracks can have messy junctions). Instead the finish line is a **point cloud
gate**: at spawn, `CarlaWorld._build_finish_line_from_kdtree()` slices every
waypoint within 20 m of the start whose offset vector is roughly perpendicular
(`|dot| < 0.15`) to the start's forward direction — that's a line across the
road. A lap completes when, after a 200 m cooldown, the car transitions from
*behind* to *in front of* that plane (a sign flip of the dot product against the
line's forward normal). **Why the cooldown and the dot-product plane:** to
reject the trivial "you start on the line" trigger and to make crossing
direction-sensitive, so reversing over the line doesn't falsely win.

---

## 5. How the sound works

`game/audio.py` runs a `sounddevice` output stream whose callback synthesizes
audio in real time on a high-priority thread.

### 5.1 The engine tone

A **sawtooth wave** whose fundamental frequency rises with speed
(`freq = 50 + speed·5`). Phase is tracked across buffers so there are no clicks
at buffer boundaries. The sawtooth is a deliberately harsh, harmonic-rich timbre
— easy to localize in the stereo field.

### 5.2 Equal-power panning

To place the sound, the RAD ratio drives left/right channel gains:

```
left_gain  = cos( (π/2) · ratio )
right_gain = sin( (π/2) · ratio )
```

**Why not just `left = 1 - ratio`, `right = ratio`?** Because of a psychoacoustic
fact (noted in `notes.txt`): linear panning makes a centered sound *quieter*
than a hard-panned one, since `0.5 + 0.5` of amplitude is less total power than
`1.0`. The trigonometric **equal-power** curve keeps perceived loudness constant
across the pan sweep (`cos² + sin² = 1`), so the driver perceives *position*
changing, not *volume*.

### 5.3 The `warp_factor` — the parameter we optimize

Before panning, the ratio is warped around center:

```
pan      = (ratio − 0.5)·2                       # remap to [-1, 1]
pan_warp = sign(pan) · |pan|^warp_factor         # exponential warp
ratio    = (pan_warp + 1)/2                       # back to [0, 1]
```

- `warp_factor = 1.0` → linear: every bit of drift moves the sound. Sensitive,
  but twitchy; no "comfortable" centered zone.
- `warp_factor = 2.0` → squared: a **safe zone** near center where small drifts
  barely move the sound, while approaching a wall ramps up sharply.
- `warp_factor ≥ 3.0` → extreme: a huge dead zone in the middle and violent
  warnings only at the very edges.

**Why this is the knob worth optimizing:** it's a pure *human-factors* tradeoff.
Too linear and the driver is overwhelmed by constant correction; too warped and
they get no warning until it's too late. There is no closed-form "best" value —
it depends on the driver, the track, and reaction time. That is *exactly* the
kind of problem Bayesian Optimization is built for.

---

## 6. The optimization: tuning the interface to the human

### 6.1 The objective

We want the `warp_factor` that minimizes **lap time** for a human driver. So the
black-box function is:

```
warp_factor  →  [human drives a lap in CARLA]  →  lap_time_seconds
```

`evaluate_my_program()` (`optimize.py`) runs each lap in a **separate process**
(`multiprocessing.Process`) with a timeout. **Why a subprocess:** a CARLA or
PyGame crash, or an unfinishable lap, must not kill the whole optimization — a
timeout/crash is just scored as a fixed penalty (`timeout + 60 s`) and the loop
continues.

### 6.2 The log-transform-and-negate trick

The raw objective is converted with `score = −log(lap_time)`:

- **`log`** compresses the heavy tail. A timed-out 360 s lap shouldn't dominate
  the GP's notion of scale versus an 80 s lap; in log-space the spread is tamer
  and closer to homoscedastic, which a Gaussian Process models far better.
- **negate** because BoTorch *maximizes* acquisition, but we want to *minimize*
  time. Maximizing `−log(time)` = minimizing time.

Reporting reverses it exactly: `time = exp(−score)`.

### 6.3 The four phases (`optimize.py::main`)

1. **Exploration** — `initial_sampling()` evaluates one **center point** plus
   `N−1` **scrambled Sobol** points. *Why Sobol:* a low-discrepancy sequence
   covers the space more evenly than uniform random, giving the surrogate a
   better global picture from few expensive human laps.
2. **Baseline** — evaluate the config's **default** value, so every reported
   improvement is measured against a meaningful reference.
3. **Bayesian Optimization** — `bayesian_optimization_loop()`:
   - Fit a **`SingleTaskGP`** (Gaussian Process) to all data so far, with input
     `Normalize` and outcome `Standardize` transforms (standard GP hygiene so
     lengthscales and noise are well-scaled).
   - Build a **`qNoisyExpectedImprovement`** acquisition function. *Why the*
     **Noisy** *variant:* the objective is genuinely noisy — the same
     `warp_factor` gives different lap times run to run because the **human is
     inconsistent**. `qNEI` accounts for observation noise rather than trusting
     each measurement as exact. (`scripts/approximate_noise.py` was built to
     *quantify* that human noise; see §7.)
   - `optimize_acqf()` proposes the next `warp_factor`; the human drives it; the
     result is appended; repeat.
4. **Reporting** — reverse the transform, compute improvement vs. default, and
   write JSON metrics, a full CSV history, and per-step surrogate plots.

### 6.4 The parameter plumbing (`opt/`)

`config/minimal.yml` declares the search space declaratively:

```yaml
parameters:
  warp_factor:
    type: float
    default: 1.0
    range: "(0.0, 6]"
```

`extract_params.py` parses the interval string — note the bracket semantics:
`(` / `)` are **exclusive** bounds and get nudged by a tiny `EPSILON`, because
BoTorch's optimizer can otherwise sample exactly on a boundary that the math
(e.g. `warp_factor = 0`) doesn't like. `params.py` then wraps each parameter
with bound-checking and provides the two conversions the loop needs:
`bounds` (the `(2, d)` tensor BoTorch wants) and `tensor_to_kwargs()` (turn a
candidate tensor back into `RacingApp(**kwargs)`). The design is **n-dimensional
ready** — today only `warp_factor` is swept, but adding more parameters to the
YAML requires no code change.

---

## 7. Dealing with the noisy human

The single biggest scientific challenge here is that **the objective is noisy
because people are inconsistent.** Drive the same setting twice and you'll get
two different lap times.

- `scripts/approximate_noise.py` measures this directly: it drives a fixed set
  of warp factors across many **shuffled** rounds (shuffling cancels
  learning/fatigue order-effects) and logs every lap time.
- `scripts/analyze_noise.py` then visualizes the spread — violin/box/strip
  plots and mean±std trend lines, in both linear and log scale — so you can
  *see* how much of the signal is real vs. noise.

These experiments justify two design choices upstream: the **log-transform**
(§6.2) and the **noisy acquisition function** (§6.3). The optimizer isn't naïvely
chasing the single fastest lap; it's estimating the warp factor with the best
*expected* performance under noise.

`scripts/replay_opt.py` closes the loop on interpretability: it re-fits the
surrogate offline from logged history and renders a 3-panel plot per step (the
noisy GP it actually used, an idealized noiseless interpolation for intuition,
and the acquisition surface) — so you can audit *why* the optimizer chose each
candidate without re-paying for human laps.

---

## 8. Why the whole thing is structured this way

| Decision | Reason |
|----------|--------|
| Synchronous CARLA mode | Deterministic, reproducible experiments. |
| KD-tree RAD over CARLA API | Order-of-magnitude faster per frame, validated equivalent by `benchmark.py`. |
| Equal-power panning | Constant perceived loudness → driver hears *position*, not *volume*. |
| `warp_factor` as the optimized knob | Pure human-factors tradeoff with no analytic optimum. |
| `−log(time)` objective | Tames the heavy tail and matches BoTorch's maximization. |
| `qNoisyExpectedImprovement` | The objective is genuinely noisy (inconsistent human). |
| Subprocess + timeout per lap | A crash or unfinishable lap can't sink the optimization. |
| Game / optimizer separation | Same code serves as both playable demo and research rig. |
| Point-cloud finish line | Robust to messy custom-track topology that CARLA's lane API mishandles. |

---

## 9. Glossary

- **RAD** — *Ratio of Available Distance*. The 0–1 lane-position cue; the core
  signal AutoRAD sonifies.
- **Warp factor** — exponent that bends the pan curve, trading a comfortable
  centered "safe zone" against sharper edge warnings; the optimized parameter.
- **Ego** — the player-controlled vehicle (CARLA terminology) and the class that
  tracks its state.
- **Surrogate / GP** — the Gaussian Process the optimizer fits to model
  `warp_factor → score` from few noisy samples.
- **Acquisition function** — the rule (here `qNoisyExpectedImprovement`) that
  picks the next point to try by balancing exploration and exploitation.
- **Sobol sequence** — a low-discrepancy sequence used to seed the search with
  even coverage of the parameter space.
