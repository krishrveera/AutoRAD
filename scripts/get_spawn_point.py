import carla
import time

def capture_transform():
    """Print the CARLA spectator camera's transform as a code snippet.

    Waits 5 seconds so you can position the spectator camera in the CARLA
    window, then reads its transform and prints a ready-to-paste
    ``carla.Transform(...)`` literal. Used to find the fixed pole-position
    spawn coordinates baked into :class:`~game.app.RacingApp`.
    """
    # 1. Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    print("--- Transform Capture Tool ---")
    print("Move your CARLA spectator camera to the desired spawn point.")
    print("Capturing in 5 seconds...")
    time.sleep(5)
    
    # 2. Get current spectator transform
    spectator = world.get_spectator()
    transform = spectator.get_transform()
    
    loc = transform.location
    rot = transform.rotation
    
    # 3. Print the code snippet for the user
    print("\n--- COPY THIS INTO YOUR CONFIG ---")
    print(f"FIXED_SPAWN_LOC = carla.Transform(")
    print(f"    carla.Location(x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}),")
    print(f"    carla.Rotation(pitch={rot.pitch:.2f}, yaw={rot.yaw:.2f}, roll={rot.roll:.2f})")
    print(f")\n")
    print("Successfully captured coordinates.")

if __name__ == "__main__":
    try:
        capture_transform()
    except Exception as e:
        print(f"Error connecting to server: {e}")