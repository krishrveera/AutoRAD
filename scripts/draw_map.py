"""
Diagnostic: draw every map waypoint as a tall pillar in the CARLA world.

Useful for verifying that a custom ``.xodr`` racetrack parsed correctly and
forms one continuous drivable loop — a low waypoint count signals a
topological break in the map. Fly the spectator camera down the track to spot
the red poles.
"""

import carla

def main():
    """Connect to CARLA, generate waypoints, and draw a red pillar at each.

    Also prints a diagnostic count of generated waypoints to flag maps with
    topological breaks.
    """
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)

    try:
        world = client.get_world()
        carla_map = world.get_map()

        print(f"Successfully connected! Loaded Map: {carla_map.name}")
        
        # Generate the waypoints
        waypoints = carla_map.generate_waypoints(distance=2.0)
        
        # DIAGNOSTIC CHECK 1: How many waypoints did CARLA actually make?
        print(f"--- DIAGNOSTIC RESULT ---")
        print(f"Total waypoints mathematically generated: {len(waypoints)}")
        
        # If this number is around 600-650, your .xodr file is perfect!
        if len(waypoints) > 500:
            print("The math is good! CARLA parsed the whole track. Drawing 15-meter pillars to find them...")
        else:
            print("CARLA gave up early. There is still a topological break in the map.")

        # 4. Draw each waypoint as a massive pillar
        for waypoint in waypoints:
            loc_bottom = waypoint.transform.location
            
            # Create a second point 15 meters straight up in the air
            loc_top = carla.Location(loc_bottom.x, loc_bottom.y, loc_bottom.z + 15.0)

            # Tell the simulator to draw a tall red line
            world.debug.draw_line(
                loc_bottom, 
                loc_top,
                thickness=0.2,                           
                color=carla.Color(r=255, g=0, b=0),  # Red
                life_time=120.0                      
            )

        print("Done! Fly your camera down the track and look for the red poles.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()