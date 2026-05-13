import pygame
import random
import sys
import time

from src.world import CarlaWorld
from src.ego import Ego
from src.render import DisplayManager
from src.audio import AudioManager
from src.controller import make_controller

class RacingApp:
    """The main application state machine and game loop."""
    
    def __init__(self, args, ctrl_kwargs):
        self.args = args
        self.ctrl_kwargs = ctrl_kwargs

        # 1. Initialize Sub-Systems
        pygame.joystick.init()
        self.world = CarlaWorld(args.host, args.port, args.delta, args.seed, args.map, args.waypoint_dist)
        self.audio = AudioManager(warp_factor=args.warp_factor)
        
        # We need the camera blueprint to know what size to make the PyGame window
        camera_bp = self.world.world.get_blueprint_library().find("sensor.camera.rgb")
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        
        self.display = DisplayManager(width=image_w, height=image_h)

        # 2. Populate the World
        vehicles = self.world.spawn_npc_traffic(args.cars, args.seed)
        if not vehicles:
            print("Error: no vehicles could be spawned. Check the map and server.")
            sys.exit(1)
            
        self.ego_vehicle = self.world.spawn_ego_vehicle()
        
        # 3. Attach Sensors and Controllers
        self.display.attach_camera(self.world.world, self.ego_vehicle)
        self.controller = make_controller(args.input, self.ego_vehicle, ctrl_kwargs)
        
        self.running = True
        self.update_gap = args.update_gap  # ms between audio updates
        self.last_update_time = time.time()

        # 4. Attach Ego states
        self.ego = Ego(
            self.ego_vehicle,
            steer_intensity=args.steer_intensity,
            time_horizon=args.time_horizon,
            dt=args.dt,
            min_speed=args.min_speed,
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