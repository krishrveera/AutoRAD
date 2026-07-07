# AutoRAD Phase 2: Knobs Implementation Plan

## Project Context

AutoRAD is a Bayesian Optimization pipeline that adapts the Racing Auditory Display (RAD) system's parameters to individual blind and low-vision (BLV) players. RAD enables BLV players to play 3D racing games (CARLA simulator) using two audio channels:

- **Sound slider**: a continuous spatial tone encoding car state (lateral position, heading, speed, track width, turn proximity) into a single auditory gestalt via stereo positioning and pitch
- **Turn indicator system (TUS)**: a series of 4 beeps triggered at distance markers ahead of upcoming turns, encoding turn timing and turn number

The BO module observes player performance and iteratively adjusts system parameters to personalize the experience per player — without the player needing to grind through sessions to "get used to" fixed settings.

**Phase 1 (completed by Nghi)**: Proof-of-concept BO module optimizing `warp_factor` only.
**Phase 2 (this plan)**: Expand the knob taxonomy, implement attentional knobs, enrich the objective function with player feedback, and move toward publication readiness.

---

## Critical Design Constraint: Intention Preservation

RAD's core philosophy is **equivalent access** — the blind player must play the *same game* as a sighted player, not an easier version. This constrains which parameters the BO is allowed to optimize.

### The Two-Layer Test

Every parameter must be classified before it can be added to the optimization:

**Interface layer (DEFENSIBLE)**: Parameters that shape how the player perceives and interacts with the game, without altering what the game demands of them. Analogy: adjusting your monitor brightness or controller sensitivity — sighted players already personalize these freely.

**Game layer (INDEFENSIBLE)**: Parameters that change what the game demands of the player. Analogy: making the font bigger on every sign in the city instead of getting the player the right glasses prescription.

**Rule**: The BO module must only optimize interface-layer parameters. Game-layer parameters must be excluded from the YAML configuration template.

---

## Knob Taxonomy

### 1. Interface Knobs (Implemented — Phase 1)

These are the existing parameters exposed in the codebase. Classification:

#### Defensible (include in BO optimization):
```python
warp_factor        # Sonification mapping — how game state maps to audio. Currently the only knob being optimized.
steer_increment    # Controller sensitivity — input mapping
steer_max          # Max steering angle — input mapping
steer_decay        # Steering return-to-center rate — input mapping
brake_increment    # Brake sensitivity — input mapping
```

#### Indefensible (exclude from BO optimization):
```python
speed_limit                  # Caps car performance — game layer
throttle_max                 # Caps car performance — game layer
throttle_increment           # Game mechanic — alters car behavior
throttle_burst               # Game mechanic — alters car behavior
throttle_burst_threshold     # Game mechanic — alters car behavior
reverse_throttle_increment   # Game mechanic — alters car behavior
```

**Implementation note**: The indefensible knobs should remain as fixed configuration but must NOT appear in the YAML template passed to the BO module. They are set once by the practitioner and left constant across all optimization iterations.

---

### 2. Attentional Knobs (To Be Implemented — Phase 2)

#### Concept

RAD has two simultaneous audio channels (sound slider + TUS). Currently, their relative prominence is fixed — the player hears both at equal salience at all times.

A sighted player doesn't experience visual information this way. Approaching a curve, the track geometry dominates their visual field. On a straightaway, their speedometer and lane position are what they monitor. The visual design naturally shifts what's perceptually dominant based on context.

Attentional knobs replicate this contextual emphasis for the audio channel — they are **meta-knobs** that modulate interface knobs based on game context.

#### Parameter: `attentional_shift_strength`

```python
attentional_shift_strength  # range: [0.0, 1.0]
# 0.0 = current RAD behavior — both channels at constant relative volume (flat mix)
# 1.0 = strong contextual emphasis — channel salience shifts based on game state
```

**Behavior**:
- When the car is on a **straightaway**: the sound slider channel is more prominent (speed/position info matters most)
- When the car is **in a turn** (TUS beeps are already firing): the TUS channel is more prominent (turn navigation matters most)

**Why this preserves intention**: The game state is unchanged. The turn indicators fire at the same distance markers. The sound slider encodes the same information. Only the *relative mix* between channels shifts — replicating the attentional affordance that visual track geometry gives sighted players for free.

**Why this is game-agnostic**: In any accessible game, there are multiple information channels. The attentional_shift_strength parameter controls how aggressively the system rebalances emphasis between *whatever channels exist* based on *whatever context is relevant*. The channels and contexts change per game; the meta-knob mechanism is identical.

#### Implementation Requirements

1. **Context detection**: The system needs to know the current game context (straightaway vs. turn zone). In RAD, this can be derived from:
   - Proximity to next turn marker (already tracked by the TUS)
   - Whether TUS beeps are currently active

2. **Gain modulation**: Given the detected context and the current `attentional_shift_strength` value, compute per-channel gain multipliers:
   ```python
   # Pseudocode — actual implementation depends on RAD's audio architecture
   def compute_channel_gains(context: str, shift_strength: float) -> dict:
       """
       Returns gain multipliers for each audio channel.

       context: 'straightaway' | 'turn_active'
       shift_strength: 0.0 (flat mix) to 1.0 (max contextual emphasis)
       """
       if context == 'straightaway':
           sound_slider_gain = 1.0
           tus_gain = 1.0 - (shift_strength * MAX_ATTENUATION)
       elif context == 'turn_active':
           sound_slider_gain = 1.0 - (shift_strength * MAX_ATTENUATION)
           tus_gain = 1.0
       return {'sound_slider': sound_slider_gain, 'tus': tus_gain}
   ```

3. **Integration with BO module**: `attentional_shift_strength` must be added to the YAML configuration template with its range `[0.0, 1.0]` so the BO can optimize it alongside `warp_factor` and the interface knobs.

4. **Smooth transitions**: The gain shift between contexts should not be abrupt (which could startle or disorient the player). Use a sigmoid or linear ramp over a short window. **CRITICAL**: the transition itself must not function as an early warning signal for turns — the shift should only begin *when* the TUS beeps start firing, not before. If the gain transition starts before the first beep, the player learns "when the soundscape shifts, a turn is coming" — creating an implicit temporal cue that violates intention preservation.

#### `attentional_shift_onset` — OPEN RESEARCH QUESTION

This was considered as a second attentional parameter controlling how far *before* a turn the emphasis shift begins. **It is currently flagged as potentially violating intention preservation** because the mix transition itself becomes an implicit early warning signal. Do not implement this unless the team explicitly decides it's defensible after discussion. Document it as an open question.

---

## Enriched Objective Function (Architectural Change)

### Problem with Current Architecture

The original architecture includes an "LLM Support" component that tells the player what the system is doing (e.g., "we are now altering the way you hear this game"). This introduces **observer bias** — if the player knows the sound is changing, they consciously monitor the audio instead of reacting naturally. You measure their analytical listening, not their intuitive racing.

### Proposed Change: Post-Optimization Feedback

Replace the LLM Support component with a **Post-Optimization Feedback** module.

**Current loop**:
```
Player plays → performance score f(x) → BO picks next knob settings → repeat
```

**New loop**:
```
Player plays → performance score → BO picks next knob settings → player plays several rounds with new settings → POST-OPTIMIZATION FEEDBACK collected → composite score f(x) → BO picks next knob settings → repeat
```

#### What Post-Optimization Feedback captures:

After an optimization step has been applied and the player has played enough rounds to experience it naturally:

- **Comfort**: "How comfortable did the game feel to play?" (Likert scale or similar)
- **Perceived clarity**: "Could you tell what was happening on the track?" (Likert scale)
- **Effort**: "How much mental effort did it take?" (Likert scale)
- **Open-ended**: Brief qualitative input for interpretability

**Critically**: The player is NOT told that parameters changed. They are simply asked about their experience. This avoids observer bias while capturing subjective experience data.

#### Composite Objective Function

```python
def composite_score(performance_metrics: dict, player_feedback: dict, 
                    alpha: float = 0.7) -> float:
    """
    Combines objective performance with subjective player experience.

    alpha: weight on performance (0.0 = pure subjective, 1.0 = pure performance)
    
    performance_metrics: dict with game-specific performance signals
        e.g., {'lap_time': float, 'off_track_pct': float, 'smoothness': float}
    
    player_feedback: dict with post-optimization feedback
        e.g., {'comfort': int, 'clarity': int, 'effort': int}  # 1-7 Likert
    """
    perf_score = normalize_performance(performance_metrics)
    feedback_score = normalize_feedback(player_feedback)
    return alpha * perf_score + (1 - alpha) * feedback_score
```

**Note**: `alpha` itself could be a meta-parameter. Early in the optimization (exploration phase), you might weight feedback higher to avoid settling into a configuration that produces good numbers but feels awful. Later (exploitation phase), performance might matter more. This is a design decision for the team.

#### Architecture Diagram Update

In the system architecture:
- Remove "LLM Support" box and its connection to the user
- Add "Post Optimization Feedback" box connected to the user (step 6: "User Feedback")
- Add "Quantifies User Experience" box that feeds into the Score box alongside "Quantifies Performance"
- Both signals feed into Score → BO as composite f(x)

---

## YAML Configuration Template (Game-Agnostic Structure)

The YAML file is the abstraction layer that makes the pipeline game-agnostic. For RAD, it should look like:

```yaml
# AutoRAD Phase 2 Configuration Template
# Only interface-layer and attentional parameters — no game-layer knobs

interface_knobs:
  - var_name: warp_factor
    range: [0.0, 1.0]
    description: "Sonification mapping — how game state maps to audio"

  - var_name: steer_increment
    range: [0.01, 0.2]  # TBD — needs calibration
    description: "Steering input sensitivity"

  - var_name: steer_max
    range: [0.1, 1.0]  # TBD — needs calibration
    description: "Maximum steering angle"

  - var_name: steer_decay
    range: [0.0, 1.0]  # TBD — needs calibration
    description: "Steering return-to-center rate"

  - var_name: brake_increment
    range: [0.01, 0.2]  # TBD — needs calibration
    description: "Brake input sensitivity"

attentional_knobs:
  - var_name: attentional_shift_strength
    range: [0.0, 1.0]
    description: "Contextual emphasis strength between audio channels"

# Channels and context triggers — required for attentional knobs
channels:
  - name: sound_slider
    description: "Continuous tone encoding car state"
  - name: turn_indicator_system
    description: "Beep sequence for upcoming turns"

context_triggers:
  - context: straightaway
    primary_channel: sound_slider
    trigger: "No active TUS beeps"
  - context: turn_active
    primary_channel: turn_indicator_system
    trigger: "TUS beeps currently firing"

# Scoring configuration
scoring:
  performance_weight: 0.7  # alpha
  feedback_weight: 0.3     # 1 - alpha
  performance_metrics:
    - lap_time
    - off_track_percentage
    - steering_smoothness
  feedback_metrics:
    - comfort
    - clarity
    - effort
```

For a different game (e.g., blind-accessible FPS), the practitioner would define entirely different `interface_knobs`, `channels`, and `context_triggers`, but the `attentional_knobs` section and the scoring structure would remain the same — that's the game-agnostic piece.

---

## Implementation Priority (Sprint Plan)

### Sprint 1 (by 11 June 2026)
- [ ] Complete Phase 1 documentation
- [ ] Code TUS integration into the completed Phase 1 build
- [ ] Run self-experiments with TUS + BO on warp_factor

### Sprint 2 (by 18 June 2026)
- [ ] Implement `attentional_shift_strength` parameter
  - [ ] Context detection logic (straightaway vs. turn_active)
  - [ ] Gain modulation function with smooth transition (sigmoid ramp)
  - [ ] Ensure transition does NOT start before first TUS beep (intention preservation)
  - [ ] Add to YAML template and BO module
- [ ] Define metrics for evaluating attentional knob impact
- [ ] Run comparative experiments: Phase 1 only vs. Phase 1 + attentional knobs
- [ ] Further brainstorm game-agnostic pipeline structure

### Sprint 3 (by 25 June 2026)
- [ ] Implement Post-Optimization Feedback module
  - [ ] Feedback collection interface (comfort, clarity, effort scales)
  - [ ] Composite scoring function
  - [ ] Integration with BO loop
- [ ] First draft of paper structure and framing

---

## Open Questions for Team Discussion

1. **`attentional_shift_onset`**: Can we defend allowing the mix shift to begin before TUS beeps fire? Current position: no, because the transition becomes an implicit early warning. Revisit with Antti and Brian.

2. **Transition smoothness**: What ramp function and duration for the gain transition between contexts? Too fast = startling. Too slow = the shift doesn't have time to take effect on short straightaways.

3. **`MAX_ATTENUATION`**: How much should the de-emphasized channel be attenuated at `shift_strength=1.0`? Full mute is clearly wrong (loses information). 30% reduction? 50%? This may need pilot testing.

4. **Feedback frequency**: How often do we collect post-optimization feedback? Every optimization step? Every N steps? Too frequent = annoying and potentially introduces its own bias. Too infrequent = noisy signal.

5. **Interface knob ranges**: The ranges for `steer_increment`, `steer_max`, `steer_decay`, and `brake_increment` need calibration from the existing codebase. What are the current default values and what range around them is playable?

6. **Joint optimization dimensionality**: Phase 1 optimized 1 knob. Phase 2 introduces up to 6 (5 interface + 1 attentional). BO sample efficiency degrades with dimensionality. Do we optimize all jointly, or use a staged approach (optimize warp_factor first, then fix it and optimize steering knobs, etc.)?
