import numpy as np


class InfinitePlane:
    def __init__(self, normal, offset, material_index, index):
        self.normal = np.array(normal)
        self.offset = offset
        self.material_index = material_index
        self.index = index

    def intersect(self, ray):
        denominator = self.normal.dot(ray.direction)
        epsilon = 1e-6
        if np.abs(denominator) < epsilon:
            return None, self.index

        t = (-self.normal.dot(ray.origin) + self.offset) / denominator
        if t < 0:
            return None, self.index

        return t, self.index

    def get_normal(self, hit_point):
        return self.normal
<<<<<<< HEAD

    def is_on_plane(self, point):
        return np.dot(point, self.normal) == self.offset
=======
>>>>>>> bc6ea0b6b837fbf94181106e2d6c15a212dda5bc
