import carla

from carla import WeatherParameters
from pathlib import Path
from random import choice
from time import sleep
from typing import Optional
from uuid import uuid4


PEDESTRIAN_CROSSING_DEFAULT = 0.15
DEFAULT_VEHICLES = 50
DEFAULT_PEDESTRIANS = 25
CAMERA_START_DELAY = 5
FRAME_SELECTION_FREQUENCY = 60
MAP_LOAD_DELAY = 2.0
DEFAULT_OUTPUT_DIRECTORY = "./output/"

class Simulation():


    def __init__(
        self,
        run_id : Optional[str] = None,
        map : Optional[str] = None,
        output_directory : Optional[str] = None,
        vehicles : int = DEFAULT_VEHICLES,
        pedestrians : int = DEFAULT_PEDESTRIANS,
        pedestrian_cross_percentage : float = PEDESTRIAN_CROSSING_DEFAULT,
        hero_spawn_point = None,
        weather_params = None,
    ):
        if run_id is None:
            run_id = uuid4()
        self.run_id = run_id

        # Create our client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)

        # Create our world object
        self.world = self.client.get_world()

        if map is None:
            map = choice(self.client.get_available_maps())
        self.client.load_world(map)
        # Give the map a moment to load
        sleep(MAP_LOAD_DELAY)

        if weather_params == None:
            weather_params = choice([
                WeatherParameters.ClearNoon,
                WeatherParameters.CloudyNoon,
                WeatherParameters.WetNoon,
                WeatherParameters.WetCloudyNoon,
                WeatherParameters.MidRainyNoon,
                WeatherParameters.HardRainNoon,
                WeatherParameters.SoftRainNoon,
                WeatherParameters.ClearSunset,
                WeatherParameters.CloudySunset,
                WeatherParameters.WetSunset,
                WeatherParameters.WetCloudySunset,
                WeatherParameters.MidRainSunset,
                WeatherParameters.HardRainSunset,
                WeatherParameters.SoftRainSunset
            ])
        self.world.set_weather(weather_params)

        # Blueprints are things we can add to the world.
        # Request from the world what blueprints are available
        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_blueprints = self.blueprint_library.filter('vehicle')
        self.vehicle_blueprints = [blueprint for blueprint in self.vehicle_blueprints if blueprint.id not in BANNED_VEHICLES]

        self._recording = False
        if output_directory is None:
            output_directory = DEFAULT_OUTPUT_DIRECTORY
        self.output_directory = output_directory
        # Ensure that the directory is created and ready to receive data
        
        Path(f"{self.output_directory}/{self.run_id}/rgb").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_directory}/{self.run_id}/semantic").mkdir(parents=True, exist_ok=True)

        self.actors = []
        self.cameras = []

        self.default_vehicles = vehicles
        self.default_pedestrians = pedestrians
        self.default_pedestrian_cross_percentage = pedestrian_cross_percentage
        self.default_hero_spawn_point = hero_spawn_point


    def spawn_hero_vehicle(self, spawn_point = None):
        blueprint = self.blueprint_library.find(TARGET_HERO_CAR)
        # Sunburnt Orange; the color of my first Mini
        blueprint.set_attribute('color', str((204, 85, 0)))

        if spawn_point is None and self.default_hero_spawn_point is not None:
            spawn_point = self.default_hero_spawn_point
        elif spawn_point is None:
            spawn_point = choice(self.world.get_map().get_spawn_points())
        
        self.hero_vehicle = self.world.spawn_actor(blueprint, spawn_point)
        self.actors.append(self.hero_vehicle)

        self.hero_vehicle.set_autopilot(True)

    def spawn_vehicles(self, count: Optional[int] = None):
        if count is None:
            count = self.default_vehicles

        spawn_points = self.world.get_map().get_spawn_points()

        for _ in range(0, count):
            blueprint = choice(self.vehicle_blueprints)

            npc = None
            while npc is None:
                spawn_point = choice(spawn_points)
                npc = self.world.try_spawn_actor(blueprint, spawn_point)

            npc.set_autopilot(True)
            self.actors.append(npc)

    def spawn_pedestrians(
        self,
        count: Optional[int] = None,
        crossing_percentage: Optional[float] = None
    ):
        if count is None:
            count = self.default_pedestrians
        if crossing_percentage is None:
            crossing_percentage = self.default_pedestrian_cross_percentage

        walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')

        # 1. Find every possible spawn point
        spawn_points = []
        for i in range(count):
            spawn_point = carla.Transform()
            location = self.world.get_random_location_from_navigation()
            if (location != None):
                spawn_point.location = location
                spawn_points.append(spawn_point)

        # 2. For each spawn point, create a walker object
        batch_commands = []
        for spawn_point in spawn_points:
            walker_bp = choice(walker_blueprints)
            create_walker_command = carla.command.SpawnActor(walker_bp, spawn_point)
            batch_commands.append(create_walker_command)

        results = self.client.apply_batch_sync(batch_commands, True)

        # Go through the results to identify each successfully created pedestrian
        pedestrian_ids = []
        for result in results:
            if not result.error:
                pedestrian_ids.append(result.actor_id)
        
        # 3. Create a walker controller for each walker we spawned
        batch_commands = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for pedestrian in pedestrian_ids:
            batch_commands.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), pedestrian))
        
        results = self.client.apply_batch_sync(batch_commands, True)

        controller_ids = []
        for result in results:
            if not result.error:
                controller_ids.append(result.actor_id)
        
        # 4. Wait for a tick to ensure client receives the last transform of the pedestrians we have just created
        self.world.tick()

        # 5. Initialize each controller, setting the target to walk towards a target. Also set their
        # predilection towards crossing the road
        self.world.set_pedestrians_cross_factor(crossing_percentage)
        controllers = self.world.get_actors(controller_ids)
        for controller in controllers:
            # start walker
            controller.start()
            # set walk to random point
            controller.go_to_location(self.world.get_random_location_from_navigation())

        self.actors += self.world.get_actors(pedestrian_ids)
        self.actors += controllers

    def spawn_everything(self):
        try:
            self.spawn_hero_vehicle()
            self.spawn_vehicles()
            self.spawn_pedestrians()
        except Exception as e:
            raise e

    def run(self, time : int):
        try:
            sleep(time)
        finally:
            self.cleanup()
    
    def launch(self, time : int):
        # Spawn everything
        self.spawn_everything()

        # Wait for some amount of seconds to spawn the camera
        # listeners; otherwise we'll have images of cars spawning
        # in the middle of the air
        sleep(CAMERA_START_DELAY)
        self.attach_camera(self.hero_vehicle)
        self.start_recording()

        # Finally run the simulation
        self.run(time)

    def record_movie_run(self, time : int):
        # Spawn everything
        self.spawn_everything()

        # Wait for some amount of seconds to spawn the camera
        # listeners; otherwise we'll have images of cars spawning
        # in the middle of the air
        sleep(CAMERA_START_DELAY)
        self.attach_movie_camera(self.hero_vehicle)
        self.start_recording()

        # Finally run the simulation
        self.run(time)

    def attach_camera(self, vehicle):
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        camera_blueprint = self.blueprint_library.find('sensor.camera.rgb')
        camera_blueprint.set_attribute("image_size_x", "800")
        camera_blueprint.set_attribute("image_size_y", "600")
        camera_blueprint.set_attribute("motion_blur_intensity", str(0.0))
        camera_blueprint.set_attribute("blur_amount", str(0.0))
        camera = self.world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)
        camera.listen(self._rgb_camera_listener)
        self.cameras.append(camera)
        
        semantic_camera_blueprint = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_blueprint.set_attribute("image_size_x", "800")
        camera_blueprint.set_attribute("image_size_y", "600")
        semantic_camera = self.world.spawn_actor(semantic_camera_blueprint, camera_transform, attach_to=vehicle)
        semantic_camera.listen(self._semantic_camera_listener)
        self.cameras.append(semantic_camera)

    def attach_movie_camera(self, vehicle):
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        camera_blueprint = self.blueprint_library.find('sensor.camera.rgb')
        camera_blueprint.set_attribute("image_size_x", "800")
        camera_blueprint.set_attribute("image_size_y", "600")
        camera_blueprint.set_attribute("motion_blur_intensity", str(0.0))
        camera_blueprint.set_attribute("blur_amount", str(0.0))
        camera = self.world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)
        camera.listen(self._rgb_movie_camera_listener)
        self.cameras.append(camera)

    def _rgb_camera_listener(self, image):
        if (not self._recording) or (image.frame % FRAME_SELECTION_FREQUENCY != 0):
            return
        image.save_to_disk(f"{self.output_directory}/{self.run_id}/rgb/{image.frame}.png")
    
    def _semantic_camera_listener(self, image):
        if (not self._recording) or (image.frame % FRAME_SELECTION_FREQUENCY != 0):
            return
        image.save_to_disk(f"{self.output_directory}/{self.run_id}/semantic/{image.frame}.png")

    def _rgb_movie_camera_listener(self, image):
        image.save_to_disk(f"{self.output_directory}/{self.run_id}/{image.frame}.png")

    def start_recording(self):
        self._recording = True
    
    def stop_recording(self):
        self._recording = False

    def cleanup(self):
        self.stop_recording()

        for camera in self.cameras:
            camera.destroy()
        self.client.apply_batch([carla.command.DestroyActor(actor) for actor in self.actors])


# We do not want to confuse the network with people on vehicles, so we'll ignore bikes
# for now
BANNED_VEHICLES = [
    "vehicle.bh.crossbike",
    "vehicle.vespa.zx125",
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.yamaha.yzf",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets"
]

# Some vehicles are too bit to use with our default camera transform. The result of this
# is that the camera captures images from inside the vehicle, which occludes what we want
# to see. I don't want to further limit vehicular diversity, however, so I will simply
# not add cameras when the vehicle is of this type
NO_CAMERA_LIST = [
    "vehicle.ford.ambulance",
    "vehicle.carlamotors.firetruck",
]

# Some vehicles are shaped such that they'll occlude our camera. Instead of creating
# custom camera configs for each blueprint, we'll just hardcode what kind of car we'll
# be driving around
TARGET_HERO_CAR = "vehicle.mini.cooper_s"