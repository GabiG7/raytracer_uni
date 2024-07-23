import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray


def parse_scene_file(file_path):
    objects = []
    index_counter = 0  # To keep track of objects list indexes
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10], index_counter)
                objects.append(material)
                index_counter += 1
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]), index_counter)
                objects.append(sphere)
                index_counter += 1
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]), index_counter)
                objects.append(plane)
                index_counter += 1
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]), index_counter)
                objects.append(cube)
                index_counter += 1
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8], index_counter)
                objects.append(light)
                index_counter += 1
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/trial.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene_file', type=str, help='Path to the scene file', default="scenes/simple_pool.txt")
    parser.add_argument('--output_image', type=str, help='Name of the output image file', default="output/trial.png")
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer


    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
