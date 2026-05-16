import carla
import numpy as np
import random
from scipy.spatial import cKDTree

NPC_MODELS = [
    "dodge", "audi", "model3", "mini", "mustang",
    "lincoln", "prius", "nissan", "crown", "impala",
]

class CarlaWorld:
    """Manages the CARLA server connection, actor spawning, and simulation ticking."""
    
    def __init__(self, host: str, port: int, delta: float, seed: int, map: str, waypoint_dist: float = 0.1):
        print("Connecting to CARLA server...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        #self.client.load_world(map)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        # 1. Immediate cleanup on connection
        self._clean_ghost_actors()
        
        # 2. Setup Synchronous Mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta
        self.world.apply_settings(settings)

        # 3. Setup map for local nearest waypoint calculation
        waypoint_arr, waypoint_tree = self._fit_waypoints_map(waypoint_dist)
        self.waypoint_array = waypoint_arr
        self.waypoint_tree = waypoint_tree

        # 4. Setup Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(seed)
        
        # 5. Build the Finish Line Gate and Racing Grid
        self.finish_line_wps = None
        spawn_points = self.map.get_spawn_points()
        start_transform = random.choice(spawn_points)
        self._build_finish_line_from_kdtree(start_transform, waypoint_arr, waypoint_tree)
    
    def _clean_ghost_actors(self):
        """Sweeps up disconnected actors from previous crashed sessions."""
        actors = self.world.get_actors()
        count = 0
        for actor in actors.filter('vehicle.*'):
            if actor.is_alive:
                actor.destroy()
                count += 1
        for actor in actors.filter('sensor.*'):
            if actor.is_alive:
                actor.destroy()
                count += 1
        if count > 0:
            print(f"Swept {count} ghost actors off the map.")

    def _fit_waypoints_map(self, waypoint_dist):
        # Load all waypoints in the map
        waypoints = self.map.generate_waypoints(waypoint_dist)
        print(f"Generated {len(waypoints)} waypoints in the map.")

        data = []
        for wp in waypoints:
            if wp.lane_type == carla.LaneType.Driving:
                data.append([
                    wp.transform.location.x, 
                    wp.transform.location.y, 
                    wp.lane_width
                ])

        # Convert the list of waypoints to a NumPy array
        waypoint_array = np.array(data)
        print("Converted driving waypoints coordinates and lane widths to NumPy array.")

        # Build a KD-tree for efficient nearest neighbor search
        waypoint_tree = cKDTree(waypoint_array[:, :2])  # Use only x and y for the tree
        print("KD-tree built for waypoint coordinates.")

        return waypoint_array, waypoint_tree
    
    def _build_finish_line_from_kdtree(self, start_transform, waypoint_array, waypoint_tree):
        """
        Builds a multi-lane finish line gate using ONLY the KD-Tree point cloud,
        completely bypassing CARLA's topological lane API.
        """
        start_loc = start_transform.location
        point = np.array([start_loc.x, start_loc.y])
        
        # 1. Get the exact forward direction of our pole position
        self.start_forward = start_transform.get_forward_vector()

        # 2. Ask the KD-Tree for EVERY waypoint within 20 meters.
        # This radius is large enough to cover a massive 8-lane highway.
        search_radius = 20.0
        candidate_indices = waypoint_tree.query_ball_point(point, r=search_radius)
        
        self.finish_line_wps = []
        
        # 3. Filter the point cloud (The Slicing Math)
        for idx in candidate_indices:
            candidate_data = waypoint_array[idx]
            
            # Create a vector pointing from the center point to the candidate point
            vec_x = candidate_data[0] - point[0]
            vec_y = candidate_data[1] - point[1]
            dist = (vec_x**2 + vec_y**2)**0.5
            
            # If distance is almost 0, it's our center point. Add it immediately.
            if dist < 0.1:
                self.finish_line_wps.append(candidate_data)
                continue
                
            # Normalize the vector
            vec_x /= dist
            vec_y /= dist
            
            # Is this point parallel to our finish line?
            # If the candidate vector is perfectly perpendicular to our forward vector, 
            # the dot product will be exactly 0.0. 
            dot_product = (vec_x * self.start_forward.x) + (vec_y * self.start_forward.y)
            
            # We allow a small tolerance (e.g., 0.15) because waypoints on curves 
            # might not be perfectly mathematically straight across the road.
            if abs(dot_product) < 0.15: 
                self.finish_line_wps.append(candidate_data)
                
        print(f"KD-Tree sliced a Finish Line containing {len(self.finish_line_wps)} valid points.")

    def spawn_racing_grid(self, num_npcs, grid_spacing=8.0):
        """
        Spawns the Ego vehicle at the start line, and staggers NPCs behind it.
        Translates raw KD-Tree points back into CARLA waypoints for grid placement.
        """
        self.vehicles = [] # Keep track of all NPCs
        
        # --- Translate KD-Tree data back to CARLA Waypoints ---
        carla_finish_wps = []
        for point_data in self.finish_line_wps:
            # Create a basic location using the X and Y from our KD-Tree data.
            # Z is 0.0 because project_to_road=True will automatically find the correct height.
            loc = carla.Location(x=float(point_data[0]), y=float(point_data[1]), z=0.0)
            
            # Get the official CARLA waypoint at this spot
            wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                carla_finish_wps.append(wp)

        if not carla_finish_wps:
            print("Error: Could not project finish line points back to the CARLA map.")
            return None

        # --- 1. Spawn the Ego Vehicle (Pole Position) ---
        # We use the center-most waypoint (index 0) of our translated list
        ego_wp = carla_finish_wps[0]
        ego_transform = ego_wp.transform
        ego_transform.location.z += 0.5 # Elevate slightly to prevent ground clipping
        
        ego_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        ego_bp.set_attribute('role_name', 'hero')
        self.ego_vehicle = self.world.spawn_actor(ego_bp, ego_transform)
        print("Spawned Ego Vehicle at Pole Position.")

        # --- 2. Spawn NPC Traffic (The Grid) ---
        lane_index = 1 # Start at index 1 so we don't spawn an NPC on top of the Ego car
        current_row_distance = 0.0

        npc_blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        for i in range(num_npcs):
            if lane_index >= len(carla_finish_wps):
                lane_index = 0
                current_row_distance += grid_spacing
            
            # Pick the translated CARLA waypoint for this grid slot
            base_wp = carla_finish_wps[lane_index]
            
            # Walk backward down the track to find the grid slot
            prev_wps = base_wp.previous(current_row_distance)
            
            if not prev_wps:
                # If we hit a dead end on this specific lane, skip to the next lane
                # instead of breaking the entire loop.
                lane_index += 1
                continue
                
            grid_wp = prev_wps[0]
            spawn_transform = grid_wp.transform
            spawn_transform.location.z += 0.5
            
            # Try to spawn the NPC
            npc_bp = random.choice(npc_blueprints)
            npc = self.world.try_spawn_actor(npc_bp, spawn_transform)
            
            if npc:
                npc.set_autopilot(True) # Let them drive!
                self.vehicles.append(npc)
                
            lane_index += 1
            
        print(f"Spawned {len(self.vehicles)} NPC competitors on the grid.")
        return self.ego_vehicle

    def spawn_npc_traffic(self, max_vehicles: int, seed: int) -> list:
        """Populates the city with AI drivers."""
        random.seed(seed)
        blueprints = [
            bp for bp in self.world.get_blueprint_library().filter("*vehicle*")
            if any(model in bp.id for model in NPC_MODELS)
        ]

        spawn_points = self.world.get_map().get_spawn_points()
        count = min(max_vehicles, len(spawn_points))

        for spawn_point in random.sample(spawn_points, count):
            actor = self.world.try_spawn_actor(random.choice(blueprints), spawn_point)
            if actor is not None:
                actor.set_autopilot(True)
                self.traffic_manager.ignore_lights_percentage(actor, random.randint(0, 50))
                self.vehicles.append(actor)
                
        print(f"Spawned {len(self.vehicles)} NPC vehicles.")
        return self.vehicles

    def spawn_ego_vehicle(self) -> carla.Vehicle:
        """Selects a vehicle from the NPC pool to become the player car."""
        if not self.vehicles:
            raise RuntimeError("No vehicles spawned. Cannot assign ego vehicle.")
        
        ego = random.choice(self.vehicles)
        ego.set_autopilot(False)
        return ego

    def tick(self):
        """Advances the simulation by one physics step."""
        self.world.tick()

    def cleanup(self):
        """Restores server settings and destroys actors."""
        print("Destroying world actors...")
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        
        for v in self.vehicles:
            if v.is_alive:
                v.destroy()