import numpy as np
import vector_utils


class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius, index):
        self.position = np.array(position)
        self.color = np.array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
        self.index = index

    def get_sample_points(self, target_point, number_of_shadow_rays):
        light_direction = vector_utils.normalize_vector(self.position - target_point)
        up_vector = np.array([0, 1, 0]) if np.abs(light_direction[1]) < 0.999 else np.array([1, 0, 0])
        right_vector = vector_utils.normalize_vector(np.cross(light_direction, up_vector))
        up_vector = vector_utils.normalize_vector(np.cross(right_vector, light_direction))

        # right_vector *= self.radius
        # up_vector *= self.radius
        right_vector *= self.radius / number_of_shadow_rays
        up_vector *= self.radius / number_of_shadow_rays

        rand_offsets = np.random.rand(number_of_shadow_rays, number_of_shadow_rays, 2) - 0.5

        sample_points = []
        for i in range(number_of_shadow_rays):
            for j in range(number_of_shadow_rays):
                # u = (i + np.random.rand()) / number_of_shadow_rays - 0.5
                # v = (j + np.random.rand()) / number_of_shadow_rays - 0.5
                # sample_points.append(self.position + u * right_vector + v * up_vector)
                sample_points.append(self.position + (i + rand_offsets[i, j, 0])
                                     * right_vector + (j + rand_offsets[i, j, 1]) * up_vector)

        return np.array(sample_points)
