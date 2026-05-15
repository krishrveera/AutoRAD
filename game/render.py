import carla
import pygame
import numpy as np

class CameraSensor:
    """Attaches to a vehicle, listens to raw data, and converts it to a PyGame surface."""
    
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, width: int, height: int):
        self.surface = pygame.Surface((width, height))
        
        bp_library = world.get_blueprint_library()
        self.camera_bp = bp_library.find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x", str(width))
        self.camera_bp.set_attribute("image_size_y", str(height))
        self.camera_bp.set_attribute("fov", "90")
        
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = world.spawn_actor(self.camera_bp, camera_transform, attach_to=vehicle)
        self.camera.listen(self._parse_image)

    def _parse_image(self, image):
        """Callback to reshape raw sensor buffers into RGB surfaces."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def destroy(self):
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()


class DisplayManager:
    """Owns the PyGame window, rendering the camera feed and HUD."""
    
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode(
            (self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Racing Simulator")
        self.camera_sensor = None

        self.font = pygame.font.SysFont(None, 36)

    def attach_camera(self, world: carla.World, vehicle: carla.Vehicle):
        """Swaps the camera view to a new vehicle."""
        if self.camera_sensor:
            self.camera_sensor.destroy()
        self.camera_sensor = CameraSensor(world, vehicle, self.width, self.height)

    def render(self, trajectory_ratio=None):
        """Draws the current frame and HUD to the screen."""
        # 1. Draw the camera feed
        if self.camera_sensor and self.camera_sensor.surface is not None:
            self.display.blit(self.camera_sensor.surface, (0, 0))
        
        # 2. Draw the HUD text
        if trajectory_ratio is not None:
            # Create a text surface (Text, Antialiasing, Color RGB)
            # Yellow text stands out well against the dark asphalt
            text_str = f"Trajectory Ratio: {trajectory_ratio:.3f}"
            text_surface = self.font.render(text_str, True, (255, 255, 0))
            
            # Draw it in the top left corner (x=20, y=20)
            self.display.blit(text_surface, (20, 20))
        
        pygame.display.flip()

    def cleanup(self):
        """Frees the camera sensor before shutdown."""
        if self.camera_sensor:
            self.camera_sensor.destroy()
        pygame.font.quit()