import random

import carla


class CarlaClient:

    def __init__(
        self,
        host="localhost",
        port=2000,
        timeout=10.0
    ):
        print(
            f"Connecting to CARLA at "
            f"{host}:{port}..."
        )

        self.client = carla.Client(
            host,
            port
        )

        self.client.set_timeout(
            timeout
        )

        self.world = (
            self.client.get_world()
        )

        # CARLA actors managed by SmartDrive
        self.ego_vehicle = None
        self.npc_vehicle = None
        self.rgb_camera = None

        print("Connected to CARLA.")

        print(
            f"Map: "
            f"{self.world.get_map().name}"
        )

    # -------------------------------------------------
    # SPAWN EGO VEHICLE
    # -------------------------------------------------
    def spawn_ego_vehicle(self):

        blueprint_library = (
            self.world.get_blueprint_library()
        )

        # Prefer Tesla Model 3.
        vehicles = blueprint_library.filter(
            "vehicle.tesla.model3"
        )

        # Fallback if Tesla is unavailable.
        if not vehicles:

            vehicles = blueprint_library.filter(
                "vehicle.*"
            )

        vehicle_blueprint = vehicles[0]

        if vehicle_blueprint.has_attribute(
            "role_name"
        ):
            vehicle_blueprint.set_attribute(
                "role_name",
                "hero"
            )

        spawn_points = (
            self.world
            .get_map()
            .get_spawn_points()
        )

        if not spawn_points:

            raise RuntimeError(
                "No vehicle spawn points found."
            )

        random.shuffle(
            spawn_points
        )

        for spawn_point in spawn_points:

            vehicle = (
                self.world.try_spawn_actor(
                    vehicle_blueprint,
                    spawn_point
                )
            )

            if vehicle is not None:

                self.ego_vehicle = vehicle

                print(
                    "\nEGO VEHICLE SPAWNED"
                )

                print(
                    f"Vehicle: "
                    f"{vehicle.type_id}"
                )

                print(
                    f"Actor ID: "
                    f"{vehicle.id}"
                )

                return vehicle

        raise RuntimeError(
            "Could not spawn ego vehicle."
        )

    # -------------------------------------------------
    # SPAWN TEST NPC VEHICLE
    # -------------------------------------------------
    def spawn_npc_ahead(
        self,
        distance=12.0
    ):

        if self.ego_vehicle is None:

            raise RuntimeError(
                "Spawn ego vehicle before NPC."
            )

        blueprint_library = (
            self.world.get_blueprint_library()
        )

        npc_blueprints = (
            blueprint_library.filter(
                "vehicle.*"
            )
        )

        # Avoid using the same blueprint as ego
        # when other vehicles are available.
        filtered_blueprints = [
            bp
            for bp in npc_blueprints
            if bp.id != self.ego_vehicle.type_id
        ]

        if filtered_blueprints:

            npc_blueprints = (
                filtered_blueprints
            )

        if not npc_blueprints:

            raise RuntimeError(
                "No NPC vehicle blueprints found."
            )

        npc_blueprint = random.choice(
            npc_blueprints
        )

        ego_transform = (
            self.ego_vehicle.get_transform()
        )

        forward_vector = (
            ego_transform.get_forward_vector()
        )

        npc_location = carla.Location(
            x=(
                ego_transform.location.x
                + forward_vector.x * distance
            ),
            y=(
                ego_transform.location.y
                + forward_vector.y * distance
            ),
            z=(
                ego_transform.location.z
                + 0.5
            )
        )

        npc_transform = carla.Transform(
            npc_location,
            ego_transform.rotation
        )

        self.npc_vehicle = (
            self.world.try_spawn_actor(
                npc_blueprint,
                npc_transform
            )
        )

        if self.npc_vehicle is None:

            raise RuntimeError(
                "Could not spawn NPC vehicle ahead."
            )

        # Keep NPC stationary for the
        # perception/risk test.
        self.npc_vehicle.set_simulate_physics(
            False
        )

        print(
            "\nNPC VEHICLE SPAWNED"
        )

        print(
            f"Vehicle: "
            f"{self.npc_vehicle.type_id}"
        )

        print(
            f"Actor ID: "
            f"{self.npc_vehicle.id}"
        )

        print(
            f"Distance ahead: "
            f"{distance:.1f} m"
        )

        return self.npc_vehicle

    # -------------------------------------------------
    # SPAWN RGB CAMERA
    # -------------------------------------------------
    def spawn_rgb_camera(
        self,
        callback,
        width=1280,
        height=720,
        fov=90
    ):

        if self.ego_vehicle is None:

            raise RuntimeError(
                "Spawn ego vehicle before camera."
            )

        blueprint_library = (
            self.world.get_blueprint_library()
        )

        camera_blueprint = (
            blueprint_library.find(
                "sensor.camera.rgb"
            )
        )

        camera_blueprint.set_attribute(
            "image_size_x",
            str(width)
        )

        camera_blueprint.set_attribute(
            "image_size_y",
            str(height)
        )

        camera_blueprint.set_attribute(
            "fov",
            str(fov)
        )

        # Front windshield / dashcam position.
        camera_transform = carla.Transform(
            carla.Location(
                x=1.5,
                y=0.0,
                z=1.7
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=0.0,
                roll=0.0
            )
        )

        self.rgb_camera = (
            self.world.spawn_actor(
                camera_blueprint,
                camera_transform,
                attach_to=self.ego_vehicle
            )
        )

        self.rgb_camera.listen(
            callback
        )

        print(
            "\nRGB CAMERA SPAWNED"
        )

        print(
            f"Camera Actor ID: "
            f"{self.rgb_camera.id}"
        )

        print(
            f"Resolution: "
            f"{width}x{height}"
        )

        print(
            f"FOV: {fov}"
        )

        return self.rgb_camera

    # -------------------------------------------------
    # DESTROY ACTORS
    # -------------------------------------------------
    def destroy(self):

        # Destroy sensor first because it is
        # attached to the ego vehicle.
        if self.rgb_camera is not None:

            print(
                "\nDestroying RGB camera..."
            )

            try:
                self.rgb_camera.stop()
            except RuntimeError:
                pass

            try:
                self.rgb_camera.destroy()
            except RuntimeError:
                pass

            self.rgb_camera = None

            print(
                "RGB camera destroyed."
            )

        # Destroy NPC vehicle.
        if self.npc_vehicle is not None:

            print(
                "Destroying NPC vehicle..."
            )

            try:
                self.npc_vehicle.destroy()
            except RuntimeError:
                pass

            self.npc_vehicle = None

            print(
                "NPC vehicle destroyed."
            )

        # Destroy ego vehicle last.
        if self.ego_vehicle is not None:

            print(
                "Destroying ego vehicle..."
            )

            try:
                self.ego_vehicle.destroy()
            except RuntimeError:
                pass

            self.ego_vehicle = None

            print(
                "Ego vehicle destroyed."
            )