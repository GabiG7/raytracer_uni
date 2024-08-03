import argparse
from PIL import Image
import numpy as np

import vector_utils
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


def fix_list_indices(obj_list):
    for i, obj in enumerate(obj_list):
        obj.index = i


def generate_rays(camera, screen_center, pixel_width, pixel_height, image_width, image_height):
    rays = []

    # Compute the origin of the screen (top-left corner)
    screen_origin = screen_center - camera.camera_right_vector * (camera.screen_width / 2)\
                    - camera.up_vector * (camera.screen_height / 2)

    for i in range(image_height):
        for j in range(image_width):
            # Compute the pixel position on the screen
            pixel_center = screen_origin + camera.camera_right_vector * (j + 0.5) * pixel_width\
                           + camera.up_vector * (i + 0.5) * pixel_height

            # Compute the ray direction
            ray_direction = pixel_center - camera.position
            rays.append(Ray(camera.position, ray_direction))

    return rays


def get_ray_intersection(ray, surfaces):
    t_min = np.inf
    index_min = None
    for obj in surfaces:
        t, index = obj.intersect(ray)
        if t is not None and t < t_min:
            t_min = t
            index_min = index

    return t_min, index_min


def get_pixel_color(ray, intersection, materials, lights, camera, background_color):

    pixel_color = np.zeros(3)
    distance, material_index = intersection
    if distance is None:
        return background_color
    # TODO check if light hits the object

    hit_point = ray.get_point_at_distance(distance)
    normal = materials[material_index].get_normal(hit_point)
    camera_point_vector = vector_utils.normalize_vector(camera.position - hit_point)

    for light in lights:
        ambient_color = materials[material_index].diffuse_color * light.color

        light_point_vector = vector_utils.normalize_vector(light.position - hit_point)
        diffuse_color = materials[material_index].diffuse_color * light.diffuse_color * \
                        np.dot(normal, light_point_vector)

        reflection_point_vector = vector_utils.normalize_vector(2 * np.dot(normal, light_point_vector) * normal
                                                                - light_point_vector)
        specular_color = materials[material_index].specular_color * light.specular_color * \
                         (np.dot(reflection_point_vector, camera_point_vector) ** materials[material_index].shininess)
        light_color = ambient_color + diffuse_color + specular_color
        pixel_color += light_color
        # TODO call shadow functions

    return light_color


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/trial.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene_file', type=str, help='Path to the scene file', default="scenes/simple_pool.txt")
    parser.add_argument('--output_image', type=str, help='Name of the output image file', default="output/trial.png")
    # parser.add_argument('--width', type=int, default=500, help='Image width')
    # parser.add_argument('--height', type=int, default=500, help='Image height')
    parser.add_argument('--width', type=int, default=100, help='Image width')
    parser.add_argument('--height', type=int, default=100, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    lights = [obj for obj in objects if type(obj) is Light]
    fix_list_indices(lights)
    materials = [obj for obj in objects if type(obj) is Material]
    fix_list_indices(materials)
    surfaces = [obj for obj in objects if type(obj) in [Cube, Sphere, InfinitePlane]]
    fix_list_indices(surfaces)
    for surface in surfaces:
        surface.material_index = objects[surface.material_index].index

    direction_ray = Ray(camera.position, camera.camera_forward_vector)
    aspect_ratio = float(args.width) / args.height
    camera.screen_height = camera.screen_width / aspect_ratio

    # Compute the screen center position
    screen_center = direction_ray.get_point_at_distance(camera.screen_distance)

    # Compute the pixel size
    pixel_width = camera.screen_width / args.width
    pixel_height = camera.screen_height / args.height  # Assuming square pixels

    # get the rays through all the pixels
    pixel_rays = generate_rays(camera, screen_center, pixel_width, pixel_height, args.width, args.height)
    rays_intersections = []
    for ray in pixel_rays:
        rays_intersections.append(get_ray_intersection(ray, surfaces))

    # TODO: Implement the ray tracer

    ray_colors = []
    for i in range(len(pixel_rays)):
        ray_colors.appends(get_pixel_color(pixel_rays[i], rays_intersections[i], materials, lights, camera,
                                           scene_settings.background_color))

    # Dummy result
    image_array = np.zeros((100, 100, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
