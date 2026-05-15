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
        self.track_length = self._calculate_track_length(self.map.get_waypoint(self.world.get_spawn_points()[0].location), waypoint_dist)
        
        # 4. Setup Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(seed)
        
        self.vehicles = []

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
    
    def _calculate_track_length(self, start_waypoint, waypoint_dist):
        """Walks a single specific lane to calculate true track length."""
        total_length = 0.0
        current_wp = start_waypoint
        
        # Lock onto the lane we started in
        target_lane_id = current_wp.lane_id 
        
        for _ in range(5000): 
            # Get all possible next waypoints
            next_wps = current_wp.next(waypoint_dist)
            
            if not next_wps:
                break # Reached a dead end
                
            # --- THE FIX: Filter out horizontal waypoints ---
            # Only keep the waypoint that stays in our specific lane
            valid_wps = [wp for wp in next_wps if wp.lane_id == target_lane_id]
            
            if valid_wps:
                next_wp = valid_wps[0]
            else:
                # Fallback: if the lane ends or merges, just take the first available
                next_wp = next_wps[0] 
                target_lane_id = next_wp.lane_id # Update our lock to the new lane
            
            # Calculate distance
            loc1 = current_wp.transform.location
            loc2 = next_wp.transform.location
            dist = ((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)**0.5
            total_length += dist
            
            # Check for completed loop
            dist_to_start = ((loc2.x - start_waypoint.transform.location.x)**2 + 
                             (loc2.y - start_waypoint.transform.location.y)**2)**0.5
            
            if total_length > 100.0 and dist_to_start < waypoint_dist:
                break
                
            current_wp = next_wp
            
        return total_length

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