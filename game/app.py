import pygame
import random
import sys
import time

from game.world import CarlaWorld
from game.ego import Ego
from game.render import DisplayManager
from game.audio import AudioManager
from game.controller import make_controller

class RacingApp:
    """The main application state machine and game loop."""
    
    def __init__(self, 
            host="127.0.0.1",  # CARLA server hostname or IP  
            port=2000,         # CARLA server port
            cars=1,            # Maximum number of NPC vehicles to spawn
            waypoint_dist=0.1, # Distance between waypoints when generating the map KDTree
            seed=0,            # Random seed for reproducible NPC behaviour
            delta=0.05,        # Fixed simulation timestep in seconds
            map="Town04",      # CARLA map to load (e.g. Town01, Town02, etc.)
            update_gap=50,     # Minimum ms between audio updates

            steer_intensity=0.1, # How much the steering angle affects the audio panning
            time_horizon=120.0,  # How far into the future the Ego class should plan its trajectory (seconds)
            dt=0.1,             # Timestep between Ego trajectory updates (seconds)
            min_speed=0.5,      # Minimum speed (m/s) before the Ego class starts updating the RAD ratio
            warp_factor=1.0,    # Exponent for warping the pan ratio to make it more perceptually linear. 1.0 = Linear (No safe zone), 2.0 = Squared (Standard safe zone), 3.0+ = Extreme (Massive safe zone, violent edge warnings)

            throttle_max=1.0,              # Maximum throttle value
            throttle_increment=0.0001,     # Incremental throttle change per input event
            throttle_burst=0.01,           # Additional throttle applied when burst threshold is exceeded
            throttle_burst_threshold=0.2,  # Speed threshold (m/s) for applying throttle burst
            speed_limit=5.0,              # Maximum speed (m/s)
            reverse_throttle_increment=0.1,# Incremental throttle change when reversing
            brake_increment=0.3,          # Incremental brake change per input event
            steer_increment=0.001,        # Incremental steering change per input event
            steer_max=0.1,                # Maximum steering angle
            steer_decay=0.2,              # Rate at which steering returns to center when no
        ):

        # 1. Initialize Sub-Systems
        pygame.joystick.init()
        self.world = CarlaWorld(
            host=host,
            port=port,
            delta=delta,
            seed=seed,
            map=map,
            waypoint_dist=waypoint_dist,
        )
        self.audio = AudioManager(warp_factor=warp_factor)
        
        # We need the camera blueprint to know what size to make the PyGame window
        camera_bp = self.world.world.get_blueprint_library().find("sensor.camera.rgb")
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        
        self.display = DisplayManager(width=image_w, height=image_h)

        # 2. Populate the World
        self.ego_vehicle = self.world.spawn_racing_grid(num_npcs=cars, grid_spacing=8.0)
        
        # 3. Attach Sensors and Controllers
        control_kwargs = dict(
            throttle_max=throttle_max,
            throttle_increment=throttle_increment,
            throttle_burst_increment=throttle_burst,
            throttle_burst_threshold=throttle_burst_threshold,
            reverse_throttle_increment=reverse_throttle_increment,
            speed_limit=speed_limit,
            brake_increment=brake_increment,
            steer_increment=steer_increment,
            steer_max=steer_max,
            steer_decay=steer_decay,
        ) 
        self.display.attach_camera(self.world.world, self.ego_vehicle)
        self.controller = make_controller(input, self.ego_vehicle, control_kwargs)
        
        self.running = True
        self.update_gap = update_gap  # ms between audio updates
        self.last_update_time = time.time()

        # 4. Attach Ego states
        self.ego = Ego(
            self.ego_vehicle,
            finish_line_wps=self.world.finish_line_wps,
            start_forward=self.world.start_forward,
            steer_intensity=steer_intensity,
            time_horizon=time_horizon,
            dt=dt,
            min_speed=min_speed,
        )
    def run(self):
        """The main game loop."""
        try:
            self.audio.start()
            while self.running:
                # 1. Advance Physics
                self.world.tick()

                # 2. Process Input Actions
                self.controller.process_control()
                self._handle_events()

                # 3. Process Audio Normalization
                current_time = time.time()
                if (current_time - self.last_update_time) * 1000 >= self.update_gap:
                    self.last_update_time = current_time
                    self.ego.update(self.world.waypoint_array, self.world.waypoint_tree)
                    self.audio.update_state(speed=self.ego.current_speed, ratio=self.ego.current_ratio)

                # 4. Render Visuals
                self.display.render(trajectory_ratio=self.ego.current_ratio)

                # 5. Determine if the vehicle has finished the track
                if self.ego.is_finished:
                    print("Congratulations! You've completed the track.")
                    self.running = False

        finally:
            self.teardown()

    def _handle_events(self):
        """Parses PyGame events like quitting, controlling, or swapping cars."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # Pass event to the keyboard/gamepad controller
            self.controller.parse_control(event)

            # Handle TAB key to switch vehicles
            if event.type == pygame.KEYUP and event.key == pygame.K_TAB:
                self._switch_vehicle()

    def _switch_vehicle(self):
        """Returns the current car to AI control and possesses a new one."""
        self.ego_vehicle.set_autopilot(True)
        
        # Pick a new random vehicle from the world's list
        self.ego_vehicle = random.choice(self.world.vehicles)
        
        if self.ego_vehicle.is_alive:
            self.ego_vehicle.set_autopilot(False)
            
            # Reattach the camera and controller to the new vehicle
            self.display.attach_camera(self.world.world, self.ego_vehicle)
            self.controller = make_controller(self.args.input, self.ego_vehicle, self.ctrl_kwargs)

    def teardown(self):
        """Safely shuts down all managers."""
        print("\nEnding game... triggering teardown sequence.")
        self.audio.stop()
        self.display.cleanup()
        self.world.cleanup()
        print(f"Maximum speed achieved during session: {self.ego.max_speed:.2f} m/s ({self.ego.max_speed * 3.6:.2f} km/h)")
        pygame.quit()
        print("Application closed gracefully.")