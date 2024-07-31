import numpy as np


class Sphere:
    def __init__(self, position, radius, material_index, index):
        self.position = np.array(position)
        self.radius = radius
        self.material_index = material_index
        self.index = index

    def intersect(self, ray):
        # solving a square equation
        b = 2 * ray.direction.dot(ray.origin - self.position)
        c = ray.origin.dot(ray.origin) + self.position.dot(self.position) - 2 * self.position.dot(ray.origin)\
            - (self.radius * self.radius)
        discriminant = (b ** 2) - (4 * c)

        # ray doesn't intersect with the camera
        if discriminant < 0:
            return None, self.index

        sqrt_res = np.sqrt(discriminant)
        t0 = (-b - sqrt_res) / 2
        t1 = (-b + sqrt_res) / 2

        # case where the object is behind the camera
        if t0 < 0 and t1 < 0:
            return None, self.index

        return min(t0, t1), self.index
