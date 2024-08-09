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
    screen_origin = screen_center - camera.camera_right_vector * (camera.screen_width / 2) \
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


def generate_shadow_ray(intersection_point, light_position):
    # Direction of shadow ray
    direction_from_light = intersection_point - light_position

    # Normalize direction vector
    direction_from_light = vector_utils.normalize_vector(direction_from_light)

    return Ray(light_position, direction_from_light)


def calculate_light_intensity(intersection_point, current_surface_index, light, shadow_rays_number, surfaces):
    # Get light plane samples by light radius and normal
    sample_points = light.get_sample_points(intersection_point, shadow_rays_number)
    total_samples = len(sample_points)
    rays_that_hit_the_surface = 0

    for sample_point in sample_points:
        shadow_ray = generate_shadow_ray(intersection_point, sample_point)

        shadow_ray_distance, shadow_ray_surface_index = get_ray_intersection(shadow_ray, surfaces)
        if shadow_ray_surface_index == current_surface_index:
            rays_that_hit_the_surface += 1

    # Calculate the fraction of light that is not obstructed
    unobstructed_fraction = rays_that_hit_the_surface / total_samples

    # Calculate the total contribution of the light source
    light_intensity = unobstructed_fraction * light.shadow_intensity + (1 - light.shadow_intensity)

    return light_intensity


def get_pixel_color(ray, intersection, surfaces, materials, lights, camera, scene_settings):
    pixel_color = np.zeros(3)
    distance, surface_index = intersection

    if surface_index is None:
        return scene_settings.background_color
    current_surface = surfaces[surface_index]
    current_material = materials[current_surface.material_index]

    # coordinates of the point where the pixel ray hits the surface
    pixel_ray_to_intersection = ray.get_point_at_distance(distance)

    # the normal to the hit point on the surface
    normal = current_surface.get_normal(pixel_ray_to_intersection)

    # ray from the camera to the surface intersection point
    intersection_point_to_camera_vector = vector_utils.normalize_vector(camera.position - pixel_ray_to_intersection)

    # seems like the ambient color is not needed
    # ambient_color = np.zeros(3)

    diffuse_color = np.zeros(3)
    specular_color = np.zeros(3)

    for light in lights:

        light_intensity = calculate_light_intensity(pixel_ray_to_intersection, surface_index, light,
                                                    scene_settings.root_number_shadow_rays, surfaces)

        # check if the light hits the current surface
        # light_ray_to_object_direction = pixel_ray_to_intersection - light.position
        # light_ray_to_object = Ray(light.position, light_ray_to_object_direction)
        # light_object_ray_distance, light_object_ray_surface_index = get_ray_intersection(light_ray_to_object,
        #                                                                                 surfaces)
        # if light_object_ray_surface_index != surface_index:
        #    continue

        # ambient color is just the color of the material multiplied by the light
        # ambient_color += current_material.diffuse_color * light.color

        # ray between the light and the intersection point on the surface
        intersection_point_to_light_vector = vector_utils.normalize_vector(light.position - pixel_ray_to_intersection)

        # calculate the diffuse color of the pixel
        current_light_diffuse = current_material.diffuse_color * light.color * np.dot(
            normal, intersection_point_to_light_vector)
        diffuse_color += current_light_diffuse * light_intensity

        # reflection ray to the vector from light to surface
        reflection_point_vector = vector_utils.normalize_vector(2 * np.dot(normal,
                                                                           intersection_point_to_light_vector) * normal
                                                                - intersection_point_to_light_vector)
        current_light_specular = current_material.specular_color * light.specular_intensity * (np.dot(
            reflection_point_vector, intersection_point_to_camera_vector) ** current_material.shininess)
        specular_color += current_light_specular * light_intensity

    light_color = diffuse_color + specular_color
    pixel_color += (current_material.transparency * scene_settings.background_color) + \
                   ((1 - current_material.transparency) * light_color) + current_material.reflection_color

    return pixel_color


def save_image(image_array):
    # image = Image.fromarray(np.uint8(image_array))
    image_array = image_array * 255
    image = Image.fromarray(image_array.astype('uint8'))

    # Save the image to a file
    image.save("scenes/trial.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene_file', type=str, help='Path to the scene file', default="scenes/sample_pool.txt")
    parser.add_argument('--output_image', type=str, help='Name of the output image file', default="output/trial.png")
    # parser.add_argument('--width', type=int, default=300, help='Image width')
    # parser.add_argument('--height', type=int, default=300, help='Image height')
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
        surface.material_index = objects[surface.material_index - 1].index

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

    counter = 0
    for ray in pixel_rays:
        rays_intersections.append(get_ray_intersection(ray, surfaces))
        counter += 1
        if counter % 1000 == 0:
            print(f"counter = {counter}")

    ray_colors = []
    counter = 0
    for i in range(len(pixel_rays)):
        ray_colors.append(get_pixel_color(pixel_rays[i], rays_intersections[i], surfaces, materials, lights, camera,
                                          scene_settings))
        counter += 1
        if counter % 1000 == 0:
            print(f"counter = {counter}")

    # Dummy result
    image_array = np.zeros((args.width, args.height, 3))
    for i in range(args.height):
        for j in range(args.width):
            image_array[i, j] = np.clip(ray_colors[i * args.width + j], 0, 1)

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
