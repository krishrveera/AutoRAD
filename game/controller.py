import carla
import pygame


class ControlObject:
    """
    Base class for vehicle control objects.

    Subclasses must override ``parse_control``; calling it on this base class
    directly raises ``NotImplementedError``.

    Parameters
    ----------
    veh : carla.Vehicle
        The vehicle to control.
    throttle_max : float
        Maximum throttle value applied to the vehicle, 0–1. Acts as a top-end
        cap for both keyboard and gamepad. For gamepad, the stick position is
        scaled by this value so full stick = throttle_max.
    throttle_increment : float
        How much throttle increases per tick during the normal ramp phase.
        Applies to both keyboard and gamepad.
    throttle_burst_threshold : float
        While throttle is below this value, the faster burst increment is used.
        Applies to both keyboard and gamepad.
    throttle_burst_increment : float
        Fast increment applied from rest until throttle_burst_threshold is
        reached. Applies to both keyboard and gamepad.
    speed_limit: float
        The maximum speed the vehicle can reach, in m/s. 
    reverse_throttle_increment : float
        How much throttle increases per tick when reversing.
    brake_increment : float
        How much brake pressure increases per tick.
    steer_increment : float
        How much the steer cache changes per tick when a steer input is held.
    steer_max : float
        Maximum absolute steering value (clamped).
    steer_decay : float
        Multiplicative decay applied to steer cache when no steer input is active.
    steer_deadzone : float
        Steer cache values within this range of zero are snapped to zero.
    """

    def __init__(
        self,
        veh,
        throttle_max: float = 1.0,
        throttle_increment: float = 0.01,
        throttle_burst_threshold: float = 0.3,
        throttle_burst_increment: float = 0.05,
        speed_limit: float = 5.0,
        reverse_throttle_increment: float = 0.1,
        brake_increment: float = 0.3,
        steer_increment: float = 0.03,
        steer_max: float = 0.7,
        steer_decay: float = 0.2,
        steer_deadzone: float = 0.01,
    ):
        self._vehicle = veh
        self._throttle = False   # True | False (keyboard) or float 0–1 (gamepad)
        self._brake = False
        self._steer = None
        self._steer_cache = 0.0
        self._control = carla.VehicleControl()

        self._throttle_max = throttle_max
        self._throttle_increment = throttle_increment
        self._throttle_burst_threshold = throttle_burst_threshold
        self._throttle_burst_increment = throttle_burst_increment

        self._speed_limit = speed_limit

        self._reverse_throttle_increment = reverse_throttle_increment
        self._brake_increment = brake_increment
        
        self._steer_increment = steer_increment
        self._steer_max = steer_max
        self._steer_decay = steer_decay
        self._steer_deadzone = steer_deadzone

    def parse_control(self, event):
        """
        Parse an input event and update the internal control state flags.
        Subclasses must override this method for their specific input device.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_control()"
        )

    def process_control(self):
        """Apply the current control state to the vehicle."""

        # --- Throttle ---
        # Keyboard: _throttle is True — ramp toward throttle_max.
        # Gamepad:  _throttle is a float 0–1 — ramp toward (axis * throttle_max).
        # Both devices share the same burst/increment ramp; the only difference
        # is the target ceiling: fixed throttle_max for keyboard, proportional
        # to stick position for gamepad.
        if self._throttle is True:
            target = self._throttle_max
        elif self._throttle:
            target = float(self._throttle) * self._throttle_max
        else:
            target = None

        speed = self._vehicle.get_velocity().length()
        margin = 4.0

        if target is not None:
            if speed >= self._speed_limit:
                target = 0.0
            elif speed > (self._speed_limit - margin):
                # Calculate a multiplier from 1.0 (start of margin) to 0.0 (at limit)
                easing_multiplier = (self._speed_limit - speed) / margin
                target = target * easing_multiplier

            if self._control.throttle < target:
                # Ramping UP (Accelerating)
                if self._control.throttle < self._throttle_burst_threshold:
                    self._control.throttle = min(
                        self._control.throttle + self._throttle_burst_increment, target
                    )
                else:
                    self._control.throttle = min(
                        self._control.throttle + self._throttle_increment, target
                    )
            elif self._control.throttle > target:
                # Ramping DOWN (Easing off as we hit the margin)
                # We use the same increment to smoothly let off the gas
                self._control.throttle = max(
                    self._control.throttle - self._throttle_increment, target
                )

            self._control.gear = 1
            self._control.brake = False
        elif not self._brake:
            self._control.throttle = 0.0

        # --- Brake / reverse ---
        if self._brake:
            # Switch to reverse when stationary and brake is held
            if self._vehicle.get_velocity().length() < 0.01 and not self._control.reverse:
                self._control.brake = 0.0
                self._control.gear = 1
                self._control.reverse = True
                self._control.throttle = min(
                    self._control.throttle + self._reverse_throttle_increment, 1.0
                )
            elif self._control.reverse:
                self._control.throttle = min(
                    self._control.throttle + self._reverse_throttle_increment, 1.0
                )
            else:
                self._control.throttle = 0.0
                self._control.brake = min(
                    self._control.brake + self._brake_increment, 1.0
                )
        else:
            self._control.brake = 0.0

        # --- Steering ---
        if self._steer is not None:
            self._steer_cache += self._steer_increment * self._steer
            self._steer_cache = max(-self._steer_max, min(self._steer_max, self._steer_cache))
        else:
            self._steer_cache *= self._steer_decay
            if abs(self._steer_cache) < self._steer_deadzone:
                self._steer_cache = 0.0

        self._control.steer = round(self._steer_cache, 4)

        self._vehicle.apply_control(self._control)


class KeyboardControlObject(ControlObject):
    """
    Keyboard-driven control using pygame arrow keys.

    * UP    – throttle
    * DOWN  – brake / reverse
    * LEFT  – steer left  (-1)
    * RIGHT – steer right (+1)
    * ENTER – disable autopilot
    """

    def parse_control(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self._vehicle.set_autopilot(False)
            if event.key == pygame.K_UP:
                self._throttle = True
            if event.key == pygame.K_DOWN:
                self._brake = True
            if event.key == pygame.K_RIGHT:
                self._steer = 1
            if event.key == pygame.K_LEFT:
                self._steer = -1

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                self._throttle = False
            if event.key == pygame.K_DOWN:
                self._brake = False
                # Full reset so forward throttle works immediately after reversing
                self._control.reverse = False
                self._control.throttle = 0.0
                self._control.gear = 1
            if event.key == pygame.K_RIGHT:
                self._steer = None
            if event.key == pygame.K_LEFT:
                self._steer = None


class GamepadControlObject(ControlObject):
    """
    Gamepad-driven control using a pygame joystick.

    Expected axis / button layout (Xbox-style):
    * Axis 1 (left stick Y, inverted) – throttle (push forward) / brake (pull back)
    * Axis 0 (left stick X)           – steering
    * Button 0 (A)                    – disable autopilot

    Throttle ramps toward (axis_value * throttle_max) using the same
    burst/increment system as the keyboard, so all throttle CLI parameters
    affect both devices identically. The analog stick position sets the
    *target*, not the instantaneous throttle value.

    Parameters
    ----------
    veh : carla.Vehicle
    joystick : pygame.joystick.Joystick
        An already-initialised pygame Joystick object.
    axis_deadzone : float
        Physical stick values below this magnitude are treated as zero.
    **kwargs
        Forwarded to :class:`ControlObject`.
    """

    _AXIS_THROTTLE_BRAKE = 1   # left stick Y (forward = negative on most pads)
    _AXIS_STEER = 0             # left stick X
    _BUTTON_AUTOPILOT_OFF = 0   # A button

    def __init__(self, veh, joystick, axis_deadzone: float = 0.05, **kwargs):
        super().__init__(veh, **kwargs)
        self._joystick = joystick
        self._axis_deadzone = axis_deadzone

    def parse_control(self, event):
        # --- Buttons ---
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == self._BUTTON_AUTOPILOT_OFF:
                self._vehicle.set_autopilot(False)

        # --- Axes ---
        if event.type == pygame.JOYAXISMOTION:
            # Throttle / brake (left stick Y, inverted: push forward → positive)
            if event.axis == self._AXIS_THROTTLE_BRAKE:
                value = -event.value
                if abs(value) < self._axis_deadzone:
                    self._throttle = 0.0
                    self._brake = False
                    # Reset reverse state so forward throttle works immediately
                    self._control.reverse = False
                    self._control.throttle = 0.0
                    self._control.gear = 1
                elif value > 0:
                    self._throttle = value  # stored as target float 0–1
                    self._brake = False
                else:
                    self._throttle = 0.0
                    self._brake = True

            # Steering (left stick X)
            if event.axis == self._AXIS_STEER:
                if abs(event.value) < self._axis_deadzone:
                    self._steer = None
                else:
                    self._steer = 1 if event.value > 0 else -1

# Factory function to create the appropriate ControlObject subclass based on input device
def make_controller(
    input_device: str,
    vehicle: carla.Vehicle,
    kwargs: dict,
) -> KeyboardControlObject | GamepadControlObject:
    """Construct the appropriate ControlObject for *input_device*."""
    if input_device == "keyboard":
        return KeyboardControlObject(vehicle, **kwargs)

    if pygame.joystick.get_count() == 0:
        print(
            "Warning: no gamepad detected — falling back to keyboard control.\n"
            "Plug in a controller and restart to use gamepad mode."
        )
        return KeyboardControlObject(vehicle, **kwargs)

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Using gamepad: {joystick.get_name()}")
    return GamepadControlObject(vehicle, joystick, **kwargs)