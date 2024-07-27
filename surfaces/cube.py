    import numpy as np


class Cube:
    def __init__(self, position, scale, material_index, index):
        self.position = np.array(position)
        self.scale = np.array(scale)
        self.material_index = material_index
        self.index = index

    def get_bounds(self):
        half_edge = self.scale / 2

        # Calculate the bounds of the cube assuming it is axis aligned
        lower_bound = self.position - half_edge
        upper_bound = self.position + half_edge

        return lower_bound, upper_bound

    def intersect(self, ray):
        # Get the bounds of the cube
        min_point, max_point = self.get_bounds()

        # Ray origin and direction
        origin = ray.origin
        direction = ray.direction

        # Initialize intersection range
        t_min = -np.inf
        t_max = np.inf

        # For each axis in 3D
        for i in range(3):
            if direction[i] == 0:
                # To avoid division by zero
                if origin[i] < min_point[i] or origin[i] > max_point[i]:
                    return None
            else:
                # Compute t values for the intersections on the current axis: t1 = entrance, t2 = exit
                t1 = (min_point[i] - origin[i]) / direction[i]
                t2 = (max_point[i] - origin[i]) / direction[i]

                # Ensure t1 is less than t2
                if t1 > t2:
                    t1, t2 = t2, t1

                # Update the intersection range
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

        # Checking if the intersection is valid
        if t_min <= t_max and t_max >= 0:
            # Return the nearest intersection point (non-negative t value), and object list index
            if t_min >= 0:
                return t_min, self.index
            else:
                return t_max, self.index

        # Case of no intersection
        return None, self.index

