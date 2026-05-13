import argparse
import carla
import random 
import time 
import numpy as np
import math 
from scipy.spatial import cKDTree
import time
import statistics

##################################################################
# ------------------ SHARED UTILITIES ----------------------------

def load_world():
    """Initiates the world from CARLA server."""
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    if settings.synchronous_mode:
        print("Currently running in: SYNCHRONOUS Mode (Client-driven)")
    else:
        print("Currently running in: ASYNCHRONOUS Mode (Server-driven)")

    carla_map = world.get_map()
    print("Successfully connected to CARLA and loaded the map!")

    return world, carla_map

def prepare_vehicles(world, carla_map, vehicle_filter='vehicle.tesla.model3', spawn_index=0):
    """Cleans up vehicles and create a new ego vehicle."""
    current_actors = world.get_actors()

    # Filter and destroy lingering vehicles
    for ghost_vehicle in current_actors.filter('vehicle.*'):
        ghost_vehicle.destroy()

    # Filter and destroy lingering pedestrians (optional, but recommended)
    for ghost_walker in current_actors.filter('walker.*'):
        ghost_walker.destroy()

    print("Ghost actors cleared.")

    # Get blueprints and spawn points
    blueprint_library = world.get_blueprint_library()
    try:
        vehicle_bp = blueprint_library.find(vehicle_filter)
    except IndexError:
        print(f"Warning: Vehicle '{vehicle_filter}' not found. Defaulting to Lincoln MKZ2017.")
        vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz_2017')

    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("There are no spawn points available in this map!")
    safe_index = spawn_index if spawn_index < len(spawn_points) else 0
    spawn_point = spawn_points[safe_index]

    # Spawn the vehicle and set it to drive automatically
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    print(f"Spawned {vehicle.type_id} at {spawn_point.location}")
    physics = vehicle.get_physics_control()
    wheelbase_cm = abs(physics.wheels[0].position.x - physics.wheels[2].position.x)
    wheelbase = wheelbase_cm / 100.0

    return vehicle, wheelbase

def kd_tree_autopilot(vehicle, waypoint_tree, waypoint_array, base_speed=0.5, lookahead_dist=6.0, steer_gain=1.5):
    """
    A custom, topology-independent autopilot that follows KD-Tree waypoints.
    Call this function once per frame inside your simulation loop.
    """
    transform = vehicle.get_transform()
    car_x = transform.location.x
    car_y = transform.location.y
    car_yaw = math.radians(transform.rotation.yaw)
    
    # 1. Project a "lookahead" point directly in front of the car
    # The faster you want to go, the further ahead you should look
    lookahead_x = car_x + (lookahead_dist * math.cos(car_yaw))
    lookahead_y = car_y + (lookahead_dist * math.sin(car_yaw))
    
    # 2. Ask the KD-Tree for the nearest drivable asphalt to that lookahead point
    distance, index = waypoint_tree.query([lookahead_x, lookahead_y], k=1)
    target_waypoint = waypoint_array[index]
    target_x, target_y = target_waypoint[0], target_waypoint[1]
    
    # 3. Calculate the angle between the car's current heading and the target waypoint
    dx = target_x - car_x
    dy = target_y - car_y
    target_heading = math.atan2(dy, dx)
    
    # Calculate the error (how far off the car's nose is from the target)
    heading_error = target_heading - car_yaw
    
    # Normalize the error to the range [-pi, pi] so the car doesn't spin in circles
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
    
    # 4. Apply the controls
    control = carla.VehicleControl()
    control.throttle = base_speed
    
    # Multiply the error by a "gain" to steer aggressively enough to make corners
    control.steer = max(-1.0, min(1.0, heading_error * steer_gain))
    
    vehicle.apply_control(control)
    
    # Optional: Draw a line to the target so you can watch the bot's "brain" work
    # target_loc = carla.Location(x=target_x, y=target_y, z=transform.location.z + 0.5)
    # world.debug.draw_line(transform.location, target_loc, thickness=0.1, color=carla.Color(255,0,0), life_time=0.1)

def update_spectator(spectator, vehicle):
    """Updates the spectator camera to follow the vehicle."""
    transform = vehicle.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    
    # 6 meters behind, 2.5 meters above
    x_offset = -6.0 * math.cos(yaw)
    y_offset = -6.0 * math.sin(yaw)
    
    camera_location = transform.location + carla.Location(x=x_offset, y=y_offset, z=2.5)
    camera_rotation = carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(camera_location, camera_rotation))

def calculate_rad(dist_left, dist_right, max_trajectory_length):
    """Calculates RAD ratio based on left and right distances"""
    if dist_left is None and dist_right is None:
        return 0.5
    elif dist_left is None:
        return dist_right / (dist_right + max_trajectory_length + 1)
    elif dist_right is None:
        return dist_left / (dist_left + max_trajectory_length + 1)
    elif dist_left == -1 or dist_right == -1:
        return 0.0
    elif dist_left == 0 and dist_right == 0:
        return 0.0
    return dist_left / (dist_left + dist_right)

####################################################################
# ---------------- CALCULATE WITH CARLA Python API -----------------
def predict_trajectory_to_impact_api(
        vehicle, wheelbase, carla_map, turn_direction="right", 
        steer_intensity=1.0, time_horizon=4.0, dt=0.1
    ):
    """
    Predicts a trajectory and stops exactly when it hits a road boundary.
    Returns the trajectory points and the distance traveled to impact.
    """
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    physics = vehicle.get_physics_control() 
    
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    # Return empty if stationary
    if speed < 0.1:
        return -1, []

    steer_intensity = max(0.0, min(1.0, steer_intensity))
    max_steer_rad = math.radians(physics.wheels[0].max_steer_angle)
    applied_steer_rad = max_steer_rad * steer_intensity
    
    if turn_direction.lower() == "left":
        steer_angle = -applied_steer_rad
    else: 
        steer_angle = applied_steer_rad

    x = transform.location.x
    y = transform.location.y
    z = transform.location.z 
    current_yaw = math.radians(transform.rotation.yaw)
    
    yaw_rate = (speed / wheelbase) * math.tan(steer_angle) if wheelbase > 0 else 0
    
    # Initialize variables
    trajectory = []
    accumulated_yaw = 0.0
    target_yaw_change = math.pi / 2.0  # 90 degrees max turn
    distance_traveled = 0.0
    
    steps = int(time_horizon / dt)
    
    for _ in range(steps):
        # 1. Update kinematics
        if abs(accumulated_yaw) < target_yaw_change:
            active_yaw_rate = yaw_rate
        else:
            active_yaw_rate = 0.0 
            
        x += speed * math.cos(current_yaw) * dt
        y += speed * math.sin(current_yaw) * dt
        
        current_yaw += active_yaw_rate * dt
        accumulated_yaw += active_yaw_rate * dt
        
        # Track arc length
        distance_traveled += speed * dt 
        
        current_loc = carla.Location(x=x, y=y, z=z)
        trajectory.append(current_loc)
        
        # 2. Collision / Boundary Check
        # project_to_road=False ensures it returns None if the point is off the lane
        wp = carla_map.get_waypoint(current_loc, project_to_road=False, lane_type=carla.LaneType.Driving)
        
        # If the waypoint is None, the point has crossed out of the drivable road boundaries
        if wp is None:
            break # Stop the simulation instantly

    if len(trajectory) == steps:
        distance_traveled = None
            
    return distance_traveled, trajectory

##################################################################
# ------------- CALCULATE WITH LOCAL KDTREE ----------------------
def fit_waypoints_map(carla_map, waypoint_dist = 2.0):
    """Retrieves the waypoints from map and fits to a KDTree"""
    # Load all waypoints in the map
    waypoints = carla_map.generate_waypoints(waypoint_dist)
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

def query_chunk_for_impact(
        chunk_points, waypoint_tree, waypoint_array
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

    return None # All points are valid if none are off-road

def predict_trajectory_to_impact_kdtree(
        vehicle, wheelbase, waypoint_array, waypoint_tree, turn_direction="right", 
        steer_intensity=1.0, time_horizon=4.0, dt=0.1
    ):
    """
    Predicts a trajectory and stops exactly when it hits a road boundary 
    using a pre-computed KD-Tree and offline numpy array.
    Returns the valid trajectory points (carla.Location) and the distance traveled to impact.
    """
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    physics = vehicle.get_physics_control() 
    
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    # Return empty if stationary
    if speed < 0.1:
        return -1, []

    steer_intensity = max(0.0, min(1.0, steer_intensity))
    max_steer_rad = math.radians(physics.wheels[0].max_steer_angle)
    applied_steer_rad = max_steer_rad * steer_intensity
    
    if turn_direction.lower() == "left":
        steer_angle = -applied_steer_rad
    else: 
        steer_angle = applied_steer_rad

    x = transform.location.x
    y = transform.location.y
    z = transform.location.z 
    current_yaw = math.radians(transform.rotation.yaw)
    
    yaw_rate = (speed / wheelbase) * math.tan(steer_angle) if wheelbase > 0 else 0
    
    # Initialize variables
    trajectory_locations = []
    trajectory_coords = []  # To store [x, y] for the KD-Tree query

    accumulated_yaw = 0.0
    target_yaw_change = math.pi / 2.0  # 90 degrees max turn
    
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
                
            x += speed * math.cos(current_yaw) * dt
            y += speed * math.sin(current_yaw) * dt
            
            current_yaw += active_yaw_rate * dt
            accumulated_yaw += active_yaw_rate * dt
            
            trajectory_locations.append(carla.Location(x=x, y=y, z=z))
            chunk_coords.append([x, y])
            
        # 2. Convert JUST this chunk's coordinates to numpy
        chunk_points = np.array(chunk_coords)
        
        # 3. Collision Check for this chunk
        local_impact_index = query_chunk_for_impact(chunk_points, waypoint_tree, waypoint_array)
        
        if local_impact_index is not None:
            # 4. IMPACT DETECTED! Calculate the absolute index and break early
            absolute_impact_index = chunk_start + local_impact_index
            valid_trajectory = trajectory_locations[:absolute_impact_index + 1]
            return len(valid_trajectory), valid_trajectory

    # 5. All points in all chunks are valid
    return None, trajectory_locations

###############################################################################
def run_benchmark(
        iterations=100, ticks_between_evals=10,
        waypoint_dist=0.5, 
        vehicle_filter='vehicle.tesla.model3', spawn_index=0,
        steer_intensity=0.3, time_horizon=10.0, dt=0.1,
    ):
    print(f"\n--- Starting Benchmark over {iterations} frames ---")
    world, carla_map = load_world()
    vehicle, wheelbase = prepare_vehicles(world, carla_map, vehicle_filter, spawn_index)
    waypoint_array, waypoint_tree = fit_waypoints_map(carla_map, waypoint_dist = waypoint_dist)
    spectator = world.get_spectator()
    
    time_method_1 = []
    time_method_2 = []
    rad_api_avg = 0.0
    rad_kdtree_avg = 0.0
    rad_loss = 0.0

    print("Warming up simulation (letting car accelerate)...")
    for _ in range(40): # 2 seconds of driving to get up to speed
        world.tick()
    
    for i in range(iterations):
        # Step the simulation so the car actually moves between checks
        print(f"---------- Iteration {i+1} ---------")
        for _ in range(ticks_between_evals):
            world.tick()
            kd_tree_autopilot(vehicle, waypoint_tree, waypoint_array)
            update_spectator(spectator, vehicle)

        # ---------------------------------------------------------
        # Benchmark Method 1: CARLA API
        # ---------------------------------------------------------
        start_t1 = time.perf_counter()
        # Call left and right turns to simulate a full evaluation frame
        dist_left, left_traj = predict_trajectory_to_impact_api(
            vehicle, wheelbase, carla_map, turn_direction="left",
            steer_intensity=steer_intensity, time_horizon=time_horizon, dt=dt
        )
        dist_right, right_traj = predict_trajectory_to_impact_api(
            vehicle, wheelbase, carla_map, turn_direction="right",
            steer_intensity=steer_intensity, time_horizon=time_horizon, dt=dt
        )
        max_trajectory_length = max(len(left_traj), len(right_traj))
        rad_api = calculate_rad(dist_left, dist_right, max_trajectory_length)

        end_t1 = time.perf_counter()
        
        time_method_1.append(end_t1 - start_t1)

        # ---------------------------------------------------------
        # Benchmark Method 2: KD-Tree + NumPy
        # ---------------------------------------------------------
        start_t2 = time.perf_counter()
        dist_left, left_traj = predict_trajectory_to_impact_kdtree(
            vehicle, wheelbase, waypoint_array, waypoint_tree, turn_direction="left",
            steer_intensity=steer_intensity, time_horizon=time_horizon, dt=dt
        )
        dist_right, right_traj = predict_trajectory_to_impact_kdtree(
            vehicle, wheelbase, waypoint_array, waypoint_tree, turn_direction="right",
            steer_intensity=steer_intensity, time_horizon=time_horizon, dt=dt
        )
        max_trajectory_length = max(len(left_traj), len(right_traj))
        rad_kdtree = calculate_rad(dist_left, dist_right, max_trajectory_length)

        end_t2 = time.perf_counter()
        
        time_method_2.append(end_t2 - start_t2)
        # ---------------------------------------------------------
        # Draw 3D Text HUD
        # ---------------------------------------------------------
        hud_location = vehicle.get_transform().location + carla.Location(z=3.0) # 3 meters above ground
        
        # Format the text so it's easy to read
        hud_text = f"API RAD:  {rad_api:.2f}\nKD  RAD:  {rad_kdtree:.2f}"
        
        # Calculate how long the text should live on screen before the next frame replaces it
        # (Tick delta * ticks between evaluations)
        text_lifetime = dt * ticks_between_evals 
        
        world.debug.draw_string(
            hud_location, 
            hud_text, 
            draw_shadow=True, 
            color=carla.Color(255, 0, 0), # Bright green
            life_time=text_lifetime
        )

        # ---------------------------------------------------------
        # Draw 3D Trajectories (KD-Tree Method)
        # ---------------------------------------------------------
        color_left = carla.Color(0, 255, 255)   # Cyan
        color_right = carla.Color(255, 0, 255)  # Magenta
        color_dot = carla.Color(255, 255, 255)  # White dots for contrast
        
        z_offset = 0.5 

        # Draw Left Trajectory (Lines + Dots)
        if left_traj and len(left_traj) > 1:
            for wp_idx in range(len(left_traj) - 1):
                p1 = left_traj[wp_idx]
                p2 = left_traj[wp_idx + 1]
                
                loc1 = carla.Location(x=p1.x, y=p1.y, z=p1.z + z_offset)
                loc2 = carla.Location(x=p2.x, y=p2.y, z=p2.z + z_offset)
                
                # Draw the line
                world.debug.draw_line(
                    loc1, loc2, thickness=0.1, color=color_left, life_time=text_lifetime
                )
                # Draw the dot at the start of the line segment
                world.debug.draw_point(
                    loc1, size=0.1, color=color_dot, life_time=text_lifetime
                )
            
            # Draw the final dot at the very end of the trajectory
            final_p = left_traj[-1]
            world.debug.draw_point(
                carla.Location(x=final_p.x, y=final_p.y, z=final_p.z + z_offset), 
                size=0.1, color=color_dot, life_time=text_lifetime
            )

        # Draw Right Trajectory (Lines + Dots)
        if right_traj and len(right_traj) > 1:
            for wp_idx in range(len(right_traj) - 1):
                p1 = right_traj[wp_idx]
                p2 = right_traj[wp_idx + 1]
                
                loc1 = carla.Location(x=p1.x, y=p1.y, z=p1.z + z_offset)
                loc2 = carla.Location(x=p2.x, y=p2.y, z=p2.z + z_offset)
                
                world.debug.draw_line(
                    loc1, loc2, thickness=0.1, color=color_right, life_time=text_lifetime
                )
                world.debug.draw_point(
                    loc1, size=0.1, color=color_dot, life_time=text_lifetime
                )
            
            final_p = right_traj[-1]
            world.debug.draw_point(
                carla.Location(x=final_p.x, y=final_p.y, z=final_p.z + z_offset), 
                size=0.1, color=color_dot, life_time=text_lifetime
            )

        # ---------------------------------------------------------
        # Compare deviations
        # ---------------------------------------------------------
        print(f"API RAD: {rad_api}")
        print(f"KDTree RAD: {rad_kdtree}")
        rad_loss += (rad_api - rad_kdtree)**2
        rad_api_avg += rad_api
        rad_kdtree_avg += rad_kdtree
        time.sleep(1.0)

    # --- Calculate and Print Results ---
    avg_t1 = statistics.mean(time_method_1)
    avg_t2 = statistics.mean(time_method_2)
    
    print("Results (Time to calculate BOTH left and right trajectories per frame):")
    print(f"Method 1 (CARLA API): {avg_t1:.5f} seconds per frame")
    print(f"Method 2 (KD-Tree):   {avg_t2:.5f} seconds per frame")
    
    speedup = avg_t1 / avg_t2 if avg_t2 > 0 else float('inf')
    print(f"\nKD-Tree is {speedup:.2f}x faster.")

    rad_loss /= iterations
    print(f"Deviations between KD-Tree and API method (MSE): {rad_loss:.4f}")

    rad_api_avg /= iterations
    rad_kdtree_avg /= iterations
    print(f"Average RAD value calculated by API method: {rad_api_avg:.4f}")
    print(f"Average RAD value calculated by KDTree method: {rad_kdtree_avg:.4f}")
    
    # Calculate equivalent max FPS if this was the ONLY thing running
    print(f"Theoretical Max FPS (API):  {1.0 / avg_t1:.0f} FPS")
    print(f"Theoretical Max FPS (Tree): {1.0 / avg_t2:.0f} FPS")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    run_benchmark(
        iterations=500, 
        ticks_between_evals=10,
        waypoint_dist=0.1, 
        vehicle_filter = 'vehicle.tesla.model3',
        spawn_index=10,
        steer_intensity=0.3, 
        time_horizon=120.0, 
        dt=0.1
    )