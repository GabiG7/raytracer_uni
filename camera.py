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


    def create_pixel_grid(self, width, height):
        self.aspect_ratio = float(width) / height
        self.screen_height = self.screen_width / self.aspect_ratio

        self.x = np.linspace(-self.screen_width / 2., self.screen_width / 2., width)
        self.y = np.linspace(self.screen_height / 2., -self.screen_height / 2., height)

        xx, yy = np.meshgrid(self.x, self.y)
        self.x = xx.flatten()
        self.y = yy.flatten()

    # Define the function to find the rotation matrix
    def create_rotation_matrix_from_vector(self, align_vector):
        cross_product = np.cross(self.camera_forward_vector, align_vector)
        dot_product = np.dot(self.camera_forward_vector, align_vector)
        cross_product_norm = np.linalg.norm(cross_product)
        kmat = np.array([[0, -cross_product[2], cross_product[1]],
                         [cross_product[2], 0, -cross_product[0]],
                         [-cross_product[1], cross_product[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - dot_product) / (cross_product_norm ** 2))
        self.rotation_matrix = rotation_matrix

    def transform_point_to_xy_plane(self, point):
        translated_point = point - self.position
        transformed_point = self.rotation_matrix @ translated_point
        return transformed_point
