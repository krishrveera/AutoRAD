import carla
import numpy as np

class Ego:
    def __init__(self, vehicle, track_length,
            steer_intensity=0.3, time_horizon=10.0, dt=0.1, min_speed=0.5):
        # Attach vehicle
        self.vehicle = vehicle
        self.transform = vehicle.get_transform()
        self.start_location = self.transform.location
        self.start_forward = self.transform.get_forward_vector()

        # Save variables to be updated
        self.current_x = self.transform.location.x
        self.current_y = self.transform.location.y
        self.current_speed = 0.0
        self.current_ratio = 0.0
        self.distance_traveled = 0.0
        self.max_speed = 0.0
        
        # Pre-calculate wheelbase
        physics = vehicle.get_physics_control()
        wheelbase_cm = abs(physics.wheels[0].position.x - physics.wheels[2].position.x)
        self.wheelbase = wheelbase_cm / 100.0

        # Other parameters necessary for trajectory prediction
        self.steer_intensity = steer_intensity
        self.time_horizon = time_horizon
        self.dt = dt
        self.min_speed = min_speed
        self.activation_distance = track_length * 0.8  # Start checking for finish line after 80% of the track is completed
   
    def _update_speed(self):
        """Updates current speed of vehicle."""
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        if speed > self.max_speed:
            self.max_speed = speed
        self.current_speed = speed
        return speed
    
    def _update_ratio(self, waypoint_array, waypoint_tree):
        """Updates RAD ratio of trajectories to impact."""
        if self.current_speed < self.min_speed:
            return
        
        dist_left, traj_left = self._predict_trajectory_to_impact_kdtree(
            waypoint_array, waypoint_tree,
            turn_direction="left", steer_intensity=self.steer_intensity, 
            time_horizon=self.time_horizon, dt=self.dt
        )
        dist_right, traj_right = self._predict_trajectory_to_impact_kdtree(
            waypoint_array, waypoint_tree,
            turn_direction="right", steer_intensity=self.steer_intensity, 
            time_horizon=self.time_horizon, dt=self.dt
        )
        ratio = self._calculate_rad(dist_left, dist_right, max_trajectory_length=int(self.time_horizon / self.dt))
        if ratio is not None:
            self.current_ratio = ratio

    def _update_distance_travelled(self):
        """Updates total distance travelled."""
        new_x = self.transform.location.x
        new_y = self.transform.location.y

        # Calculate delta (how far we moved this exact frame)
        delta = ((new_x - self.current_x)**2 + (new_y - self.current_y)**2)**0.5
        
        # Add the delta to the grand total
        self.distance_traveled += delta 
        
        self.current_x = new_x
        self.current_y = new_y

    def update(self, waypoint_array, waypoint_tree):
        """Updates all relevant states of the ego vehicle"""
        self.transform = self.vehicle.get_transform()
        
        self._update_speed()
        self._update_ratio(waypoint_array, waypoint_tree)
        self._update_distance_travelled()

    def _query_chunk_for_impact(
        self, chunk_points, waypoint_tree, waypoint_array
    ):
        # Query the tree with the entire batch of trajectory points simultaneously.
        # k=1 means we only want the single closest waypoint for each point.
        distances, indices = waypoint_tree.query(chunk_points, k=1)
        closest_waypoints = waypoint_array[indices]
        
        # Check if the car has gone off the lane
        lane_widths = closest_waypoints[:, 2]  # Assuming lane width is the third column

        off_road_mask = distances > (lane_widths / 2.0)
        
        if np.any(off_road_mask):
            return np.argmax(off_road_mask)

        return None
    
    def _predict_trajectory_to_impact_kdtree(
        self, waypoint_array, waypoint_tree, turn_direction="right", 
        steer_intensity=0.3, time_horizon=20.0, dt=0.1
    ):
        """
        Predicts a trajectory and stops exactly when it hits a road boundary 
        using a pre-computed KD-Tree and offline numpy array.
        Returns the valid trajectory points (carla.Location) and the distance traveled to impact.
        """
        velocity = self.vehicle.get_velocity()
        physics = self.vehicle.get_physics_control() 
        
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Return empty if stationary
        if speed < 0.1:
            return -1, []

        steer_intensity = max(0.0, min(1.0, steer_intensity))
        max_steer_rad = np.radians(physics.wheels[0].max_steer_angle)
        applied_steer_rad = max_steer_rad * steer_intensity
        
        if turn_direction.lower() == "left":
            steer_angle = -applied_steer_rad
        else: 
            steer_angle = applied_steer_rad

        x = self.transform.location.x
        y = self.transform.location.y
        z = self.transform.location.z 
        current_yaw = np.radians(self.transform.rotation.yaw)
        
        yaw_rate = (speed / self.wheelbase) * np.tan(steer_angle) if self.wheelbase > 0 else 0
        
        # Initialize variables
        trajectory_locations = []

        accumulated_yaw = 0.0
        target_yaw_change = np.pi / 2.0  # 90 degrees max turn
        
        total_steps = int(time_horizon / dt)
        steps_per_chunk = int(1.0 / dt)  # Process 1 second of trajectory at a time
        
        # 1. Generate and evaluate kinematics trajectory in chunks
        for chunk_start in range(0, total_steps, steps_per_chunk):
            chunk_coords = []
            
            # Calculate how many steps are left (prevents overshoot on the final chunk)
            current_chunk_steps = min(steps_per_chunk, total_steps - chunk_start)

            for _ in range(current_chunk_steps):
                if abs(accumulated_yaw) < target_yaw_change:
                    active_yaw_rate = yaw_rate
                else:
                    active_yaw_rate = 0.0 
                    
                x += speed * np.cos(current_yaw) * dt
                y += speed * np.sin(current_yaw) * dt
                
                current_yaw += active_yaw_rate * dt
                accumulated_yaw += active_yaw_rate * dt
                
                trajectory_locations.append(carla.Location(x=x, y=y, z=z))
                chunk_coords.append([x, y])
                
            # 2. Convert JUST this chunk's coordinates to numpy
            chunk_points = np.array(chunk_coords)
            
            # 3. Collision Check for this chunk
            local_impact_index = self._query_chunk_for_impact(chunk_points, waypoint_tree, waypoint_array)
            
            if local_impact_index is not None:
                # 4. IMPACT DETECTED! Calculate the absolute index and break early
                absolute_impact_index = chunk_start + local_impact_index
                valid_trajectory = trajectory_locations[:absolute_impact_index + 1]
                return len(valid_trajectory), valid_trajectory

        # 5. All points in all chunks are valid
        return None, trajectory_locations
    
    def _calculate_rad(self, dist_left, dist_right, max_trajectory_length):
        """Calculates RAD ratio based on left and right distances"""
        if dist_left is None and dist_right is None:
            return 0.5
        elif dist_left is None:
            return dist_right / (dist_right + max_trajectory_length + 1)
        elif dist_right is None:
            return dist_left / (dist_left + max_trajectory_length + 1)
        elif dist_left == -1 or dist_right == -1:
            return None
        return dist_left / (dist_left + dist_right)
    
    def _check_finish_line(self, waypoint_array, waypoint_tree):
        """Checks if the car mathematically pierced the finish line plane while on the correct track segment."""
        if self.distance_traveled < self.activation_distance:
            return

        current_loc = self.transform.location
        
        # 1. Create a vector from the start line to the car
        vec_x = current_loc.x - self.start_location.x
        vec_y = current_loc.y - self.start_location.y
        
        # 2. Check Direction: Are we in front of the line?
        dot_product = (vec_x * self.start_forward.x) + (vec_y * self.start_forward.y)
        is_in_front = dot_product > 0
        
        # 3. Did we cross the plane THIS exact frame?
        if self.was_behind_line and is_in_front:
            
            # 4. KD-Tree Query: Get the nearest waypoint data
            point = np.array([[current_loc.x, current_loc.y]])
            distance, index = waypoint_tree.query(point, k=1)
            closest_waypoint = waypoint_array[index[0]]
            lane_width = closest_waypoint[2] 
            
            # Check A: Track Bounds (Are we on the asphalt?)
            on_asphalt = distance[0] <= (lane_width / 2.0)
            
            # Check B: Topology Bounds
            # If we crossed the infinite plane on the other side of the map, 
            # this distance would be massive. We restrict the "Trigger Volume"
            # to a small radius around the start location. We use lane_width * 1.5 
            # to safely account for crossing near the edges or at high speeds.
            dist_to_start_center = (vec_x**2 + vec_y**2)**0.5
            at_finish_line = dist_to_start_center <= (lane_width * 1.5) 
            
            if on_asphalt and at_finish_line:
                self.is_finished = True
                
        # Save current state for the next frame's comparison
        self.was_behind_line = not is_in_front