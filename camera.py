import vector_utils
import numpy as np


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up_vector = np.array(up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.screen_height = None

        # the forward vector of the camera
        self.camera_forward_vector = vector_utils.normalize_vector(self.look_at - self.position)
        self.camera_right_vector = vector_utils.normalize_vector(np.cross(self.camera_forward_vector, self.up_vector))
        self.up_vector = vector_utils.normalize_vector(np.cross(self.camera_forward_vector, self.camera_right_vector))
