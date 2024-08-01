import numpy as np
import vector_utils


class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius, index):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
        self.index = index

    def get_sample_points(self, normal):
        # Number of points along each axis of the square
        count = int(self.radius)
        spacing = 1.0

        # Create orthogonal basis vectors for the light matrix plane
        u = np.cross(normal, [0, 1, 0])
        if np.linalg.norm(u) == 0:
            u = np.cross(normal, [1, 0, 0])
        vector_utils.normalize_vector(u)
        v = np.cross(normal, u)

        # Center the grid around the light's position
        sample_points = []
        start = -spacing * (count // 2)
        for i in range(count):
            for j in range(count):
                point = start + i * spacing * u + start + j * spacing * v
                sample_points.append(self.position + point)

        # A list of numpy arrays representing the 3D coordinates of the light sample points
        return sample_points
