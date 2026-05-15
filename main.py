"""
CARLA vehicle simulation with keyboard or gamepad control.

Usage
-----
    python main.py [options]
"""

import argparse
from game.app import RacingApp

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CARLA manual-control demo with NPC traffic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Simulation ---
    sim = parser.add_argument_group("simulation")
    sim.add_argument("--host", default="127.0.0.1", help="CARLA server hostname or IP")
    sim.add_argument("--port", type=int, default=2000, help="CARLA server port")
    sim.add_argument("--cars", type=int, default=1, help="Maximum number of NPC vehicles to spawn")
    sim.add_argument("--waypoint-dist", type=float, default=0.1, help="Distance between waypoints when generating the map KDTree")
    sim.add_argument("--seed", type=int, default=0, help="Random seed for reproducible NPC behaviour")
    sim.add_argument("--delta", type=float, default=0.05, help="Fixed simulation timestep in seconds")
    sim.add_argument("--map", default="Town04", help="CARLA map to load (e.g. Town01, Town02, etc.)")
    sim.add_argument("--update-gap", type=int, default=50, help="Minimum ms between audio updates")
    
    sim.add_argument("--steer-intensity", type=float, default=0.1, help="How much the steering angle affects the audio panning")
    sim.add_argument("--time-horizon", type=float, default=120.0, help="How far into the future the Ego class should plan its trajectory (seconds)")
    sim.add_argument("--dt", type=float, default=0.1, help="Timestep between Ego trajectory updates (seconds)")
    sim.add_argument("--min-speed", type=float, default=0.5, help="Minimum speed (m/s) before the Ego class starts updating the RAD ratio")

    sim.add_argument("--warp-factor", type=float, default=1.0, help="Exponent for warping the pan ratio to make it more perceptually linear. 1.0 = Linear (No safe zone), 2.0 = Squared (Standard safe zone), 3.0+ = Extreme (Massive safe zone, violent edge warnings)")

    # --- Input ---
    inp = parser.add_argument_group("input")
    inp.add_argument("--input", choices=["keyboard", "gamepad"], default="gamepad", help="Input device")

    # --- Control tuning ---
    ctrl = parser.add_argument_group("control tuning")
    ctrl.add_argument("--throttle-max", type=float, default=1.0, metavar="F")
    ctrl.add_argument("--throttle-increment", type=float, default=0.0001, metavar="F")
    ctrl.add_argument("--throttle-burst", type=float, default=0.01, metavar="F")
    ctrl.add_argument("--throttle-burst-threshold", type=float, default=0.2, metavar="F")
    ctrl.add_argument("--speed-limit", type=float, default=5.0, metavar="MS")
    ctrl.add_argument("--reverse-throttle-increment", type=float, default=0.1, metavar="F")
    ctrl.add_argument("--brake-increment", type=float, default=0.3, metavar="F")
    ctrl.add_argument("--steer-increment", type=float, default=0.001, metavar="F")
    ctrl.add_argument("--steer-max", type=float, default=0.1, metavar="F")
    ctrl.add_argument("--steer-decay", type=float, default=0.2, metavar="F")

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Instantiate the application and run it
    app = RacingApp(
        host=args.host,
        port=args.port,
        cars=args.cars,
        waypoint_dist=args.waypoint_dist,
        seed=args.seed,
        delta=args.delta,
        map=args.map,
        update_gap=args.update_gap,

        steer_intensity=args.steer_intensity,
        time_horizon=args.time_horizon,
        dt=args.dt,
        min_speed=args.min_speed,

        warp_factor=args.warp_factor,

        throttle_max=args.throttle_max,
        throttle_increment=args.throttle_increment,
        throttle_burst=args.throttle_burst,
        throttle_burst_threshold=args.throttle_burst_threshold,
        speed_limit=args.speed_limit,
        reverse_throttle_increment=args.reverse_throttle_increment,
        brake_increment=args.brake_increment,
        steer_increment=args.steer_increment,
        steer_max=args.steer_max,
        steer_decay=args.steer_decay,
    )
    app.run()

if __name__ == "__main__":
    main()