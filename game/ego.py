import carla
import numpy as np

class Ego:
    """Tracks the player vehicle's state and computes the RAD audio cue.

    On every update the Ego reads the vehicle's transform and velocity, then
    derives the **RAD ratio** (Ratio of Available Distance): it predicts two
    kinematic "what if I steered hard left / hard right" trajectories and
    measures how far each travels before leaving the drivable road. The ratio
    of those two distances becomes a 0–1 pan value (0 = wall close on the
    left, 1 = wall close on the right, 0.5 = centered) that the audio engine
    turns into stereo panning. The class also accumulates distance travelled,
    tracks top speed, and detects finish-line crossings to end a lap.
    """

    def __init__(self, vehicle, finish_line_wps, start_forward,
            steer_intensity=0.3, time_horizon=10.0, dt=0.1, min_speed=0.5):
        """Initialize Ego state and cache the vehicle's wheelbase.

        Parameters
        ----------
        vehicle : carla.Vehicle
            The player-controlled vehicle to track.
        finish_line_wps : list[numpy.ndarray]
            Finish-line gate points as ``[x, y, width]`` rows (from the world's
            KD-tree slice).
        start_forward : carla.Vector3D
            Forward direction of the start/finish line, used as the normal for
            the directional crossing check.
        steer_intensity : float, optional
            Fraction (0–1) of max steer angle used when projecting the
            left/right trajectories.
        time_horizon : float, optional
            How far ahead, in seconds, to project each trajectory.
        dt : float, optional
            Integration timestep, in seconds, for trajectory prediction.
        min_speed : float, optional
            Speed (m/s) below which the RAD ratio is not updated (avoids noise
            while stationary).
        """
        # Attach vehicle
        self.vehicle = vehicle
        self.transform = vehicle.get_transform()
        self.start_location = self.transform.location
        self.finish_line_wps = finish_line_wps
        self.start_forward = start_forward

        # Save variables to be updated
        self.current_x = None
        self.current_y = None
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

        # Finish line state
        self.was_behind_line = True  # Assume we start behind the line
        self.is_finished = False
   
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
        if self.current_x is None or self.current_y is None:
            self.current_x = self.transform.location.x
            self.current_y = self.transform.location.y
            return
        
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
        self._check_finish_line()

    def _query_chunk_for_impact(
        self, chunk_points, waypoint_tree, waypoint_array
    ):
        """Find the first trajectory point in a chunk that leaves the road.

        Batch-queries the KD-tree for the nearest waypoint of every point in
        the chunk, then flags any point whose distance to that waypoint
        exceeds half the local lane width (i.e. it has crossed the lane edge).

        Parameters
        ----------
        chunk_points : numpy.ndarray
            ``(k, 2)`` array of candidate ``(x, y)`` trajectory points.
        waypoint_tree : scipy.spatial.cKDTree
            KD-tree over road waypoint coordinates.
        waypoint_array : numpy.ndarray
            ``(N, 3)`` array of ``[x, y, lane_width]`` rows.

        Returns
        -------
        int | None
            The local index of the first off-road point, or ``None`` if the
            whole chunk stays on the road.
        """
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
    
    def _check_finish_line(self):
        """Checks if the car has crossed any of the horizontal finish line KD-Tree points."""
        
        # 1. Lap Cooldown (e.g., must drive at least 200 meters before winning)
        cooldown_distance = 200.0 
        if self.distance_traveled < cooldown_distance:
            return

        current_loc = self.transform.location
        
        # 2. Are we inside the Finish Line Gate?
        near_gate = False
        active_wp = None
        
        for wp in self.finish_line_wps:
            # wp is now a data array: [x, y, width]
            wp_x = wp[0]
            wp_y = wp[1]
            wp_width = wp[2]

            # Check if we are within this specific lane's width
            dist_to_wp = ((current_loc.x - wp_x)**2 + (current_loc.y - wp_y)**2)**0.5
            
            # Using wp_width as our trigger radius
            if dist_to_wp <= (wp_width * 1.0): 
                near_gate = True
                active_wp = wp
                break
                
        # 3. Directional Plane Check (The Dot Product)
        if near_gate:
            # Vector from the active KD-Tree point to the car
            vec_x = current_loc.x - active_wp[0]
            vec_y = current_loc.y - active_wp[1]
            
            # Dot product against the finish line's forward direction
            dot_product = (vec_x * self.start_forward.x) + (vec_y * self.start_forward.y)
            is_in_front = dot_product > 0
            
            if self.was_behind_line and is_in_front:
                self.is_finished = True
                
            self.was_behind_line = not is_in_front
        else:
            # If we aren't near the gate, we are safely "behind" it
            self.was_behind_line = True