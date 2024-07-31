import vector_utils


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = vector_utils.normalize_vector(direction)

    def get_point_at_distance(self, distance):
        return self.origin + distance * self.direction
