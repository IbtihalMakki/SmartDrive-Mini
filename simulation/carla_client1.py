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

        self.client.set_timeout(timeout)

        self.world = self.client.get_world()

        self.ego_vehicle = None
        self.npc_vehicle = None
        self.rgb_camera = None

        self.test_traffic_light = None
        self.test_traffic_light_group = {}
        self.test_traffic_light_state = None
        self._last_traffic_light_verification = None

        print("Connected to CARLA.")
        print(
            f"Map: {self.world.get_map().name}"
        )

    # -------------------------------------------------
    # SPAWN EGO VEHICLE
    # -------------------------------------------------

    def spawn_ego_vehicle(self):

        blueprint_library = (
            self.world.get_blueprint_library()
        )

        vehicles = blueprint_library.filter(
            "vehicle.tesla.model3"
        )

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

        random.shuffle(spawn_points)

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
    # TRAFFIC-LIGHT TEST SCENE
    # -------------------------------------------------

    def spawn_ego_for_traffic_light_test(
        self,
        distance_before_light=18.0
    ):
        """
        Spawn the ego vehicle on a driving waypoint that
        approaches a CARLA traffic light.

        The selected light is stored in self.test_traffic_light.

        IMPORTANT:
        This is deterministic test-scene setup only. The
        TrafficLightHandler still reads CARLA's semantic state.
        """

        if self.ego_vehicle is not None:
            raise RuntimeError(
                "Ego vehicle already exists."
            )

        carla_map = self.world.get_map()

        traffic_lights = list(
            self.world.get_actors().filter(
                "traffic.traffic_light*"
            )
        )

        if not traffic_lights:
            raise RuntimeError(
                "No traffic lights found in this CARLA map."
            )

        blueprint_library = (
            self.world.get_blueprint_library()
        )

        vehicles = blueprint_library.filter(
            "vehicle.tesla.model3"
        )

        if not vehicles:
            vehicles = blueprint_library.filter(
                "vehicle.*"
            )

        if not vehicles:
            raise RuntimeError(
                "No vehicle blueprints found."
            )

        vehicle_blueprint = vehicles[0]

        if vehicle_blueprint.has_attribute(
            "role_name"
        ):
            vehicle_blueprint.set_attribute(
                "role_name",
                "hero"
            )

        random.shuffle(traffic_lights)

        for traffic_light in traffic_lights:

            # CARLA traffic lights expose stop waypoints.
            try:
                stop_waypoints = (
                    traffic_light.get_stop_waypoints()
                )
            except Exception:
                stop_waypoints = []

            for stop_waypoint in stop_waypoints:

                # Walk backwards from the stop line so the ego
                # starts on the lane controlled by this light.
                previous_waypoints = (
                    stop_waypoint.previous(
                        distance_before_light
                    )
                )

                if not previous_waypoints:
                    continue

                # Prefer same road/lane as the stop waypoint.
                same_lane = [
                    waypoint
                    for waypoint in previous_waypoints
                    if (
                        waypoint.road_id
                        == stop_waypoint.road_id
                        and
                        waypoint.lane_id
                        == stop_waypoint.lane_id
                    )
                ]

                candidates = (
                    same_lane
                    if same_lane
                    else previous_waypoints
                )

                for waypoint in candidates:

                    transform = waypoint.transform
                    transform.location.z += 0.3

                    vehicle = (
                        self.world.try_spawn_actor(
                            vehicle_blueprint,
                            transform
                        )
                    )

                    if vehicle is None:
                        continue

                    self.ego_vehicle = vehicle
                    self.test_traffic_light = (
                        traffic_light
                    )

                    # The first forced state freezes the entire group.

                    print(
                        "\nTRAFFIC-LIGHT TEST EGO SPAWNED"
                    )

                    print(
                        f"Vehicle: {vehicle.type_id}"
                    )

                    print(
                        f"Actor ID: {vehicle.id}"
                    )

                    print(
                        f"Traffic Light Actor ID: "
                        f"{traffic_light.id}"
                    )

                    print(
                        f"Road ID: {waypoint.road_id}"
                    )

                    print(
                        f"Lane ID: {waypoint.lane_id}"
                    )

                    print(
                        f"Requested start distance: "
                        f"{distance_before_light:.1f} m"
                    )

                    return vehicle

        raise RuntimeError(
            "Could not create a traffic-light test scene. "
            "No free controlled-lane spawn location was found."
        )

    def set_test_traffic_light_state(
        self,
        state_name
    ):
        """
        Force every light in the selected test group to the same state.
        Verification uses subsequent semantic observations, since CARLA
        getters report the last tick rather than immediate setter results.
        """

        if self.test_traffic_light is None:
            raise RuntimeError(
                "Traffic-light test scene is not initialized."
            )

        state_map = {
            "RED": carla.TrafficLightState.Red,
            "YELLOW": carla.TrafficLightState.Yellow,
            "GREEN": carla.TrafficLightState.Green
        }

        state_name = state_name.upper()

        if state_name not in state_map:
            raise ValueError(
                f"Unsupported traffic-light state: "
                f"{state_name}"
            )

        light = self.test_traffic_light

        # Include the selected actor even if CARLA omits it from the list,
        # and deduplicate by actor ID rather than Python wrapper identity.
        group = {light.id: light}
        group.update({
            member.id: member
            for member in light.get_group_traffic_lights()
        })
        self.test_traffic_light_group = group
        self.test_traffic_light_state = None
        self._last_traffic_light_verification = None

        # Do not suppress failures: a partially forced scene is not valid.
        for member in group.values():
            member.freeze(True)
        for member in group.values():
            member.set_state(state_map[state_name])

        self.test_traffic_light_state = state_name

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"TRAFFIC-LIGHT TEST: "
            f"FORCED {state_name}"
        )

        print(
            f"Traffic Light Actor ID: "
            f"{light.id}"
        )
        print(f"Traffic Light Group IDs: {sorted(group)}")

        print(
            "=" * 70
        )

        return True

    def verify_test_traffic_light_state(self, status):
        """Check the exact ego actor/state observed by TrafficLightHandler."""
        actor_id = status["actor_id"]
        observed = status["state"]
        expected = self.test_traffic_light_state
        in_group = actor_id in self.test_traffic_light_group
        verified = expected is not None and in_group and observed == expected
        observation = (actor_id, observed, expected, in_group)

        if observation != self._last_traffic_light_verification:
            print(
                "TRAFFIC-LIGHT TEST VERIFY: "
                f"ego actor={actor_id}, expected={expected}, "
                f"observed={observed}, in_group={in_group}, "
                f"result={'MATCH' if verified else 'WAIT'}"
            )
            self._last_traffic_light_verification = observation

        return verified

    def get_test_traffic_light(self):

        return self.test_traffic_light

    # -------------------------------------------------
    # SPAWN NPC ON SAME LANE AHEAD
    # -------------------------------------------------

    def spawn_npc_ahead(
        self,
        distance=20.0
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

        filtered_blueprints = [
            bp
            for bp in npc_blueprints
            if bp.id != self.ego_vehicle.type_id
        ]

        if filtered_blueprints:
            npc_blueprints = filtered_blueprints

        if not npc_blueprints:
            raise RuntimeError(
                "No NPC vehicle blueprints found."
            )

        npc_blueprint = random.choice(
            npc_blueprints
        )

        carla_map = self.world.get_map()

        ego_location = (
            self.ego_vehicle.get_location()
        )

        ego_waypoint = (
            carla_map.get_waypoint(
                ego_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        )

        if ego_waypoint is None:
            raise RuntimeError(
                "Could not find ego road waypoint."
            )

        next_waypoints = (
            ego_waypoint.next(
                distance
            )
        )

        if not next_waypoints:
            raise RuntimeError(
                "Could not find waypoint ahead "
                "for NPC."
            )

        same_lane_waypoints = [
            waypoint
            for waypoint in next_waypoints
            if (
                waypoint.road_id
                == ego_waypoint.road_id
                and waypoint.lane_id
                == ego_waypoint.lane_id
            )
        ]

        if same_lane_waypoints:
            npc_waypoint = (
                same_lane_waypoints[0]
            )
        else:
            npc_waypoint = (
                next_waypoints[0]
            )

        npc_transform = (
            npc_waypoint.transform
        )

        npc_transform.location.z += 0.3

        self.npc_vehicle = (
            self.world.try_spawn_actor(
                npc_blueprint,
                npc_transform
            )
        )

        if self.npc_vehicle is None:
            raise RuntimeError(
                "Could not spawn NPC vehicle "
                "on waypoint ahead."
            )

        self.npc_vehicle.set_simulate_physics(
            False
        )

        true_distance = (
            self.ego_vehicle
            .get_location()
            .distance(
                self.npc_vehicle.get_location()
            )
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
            f"Road ID: "
            f"{npc_waypoint.road_id}"
        )

        print(
            f"Lane ID: "
            f"{npc_waypoint.lane_id}"
        )

        print(
            f"Requested distance ahead: "
            f"{distance:.1f} m"
        )

        print(
            f"Initial true distance: "
            f"{true_distance:.2f} m"
        )

        return self.npc_vehicle

    # -------------------------------------------------
    # MOVE NPC AWAY
    # -------------------------------------------------

    def move_npc_away(
        self,
        distance=40.0
    ):
        """
        Move the controlled NPC farther ahead.

        Used for the recovery test:

        STOP
        -> REASSESS
        -> RESUME
        """

        if self.npc_vehicle is None:

            print(
                "NPC vehicle does not exist."
            )

            return False

        carla_map = self.world.get_map()

        npc_location = (
            self.npc_vehicle.get_location()
        )

        npc_waypoint = (
            carla_map.get_waypoint(
                npc_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        )

        if npc_waypoint is None:

            print(
                "Could not find NPC waypoint."
            )

            return False

        next_waypoints = (
            npc_waypoint.next(
                distance
            )
        )

        if not next_waypoints:

            print(
                "Could not find waypoint "
                "to move NPC away."
            )

            return False

        same_lane = [
            waypoint
            for waypoint in next_waypoints
            if (
                waypoint.road_id
                == npc_waypoint.road_id
                and waypoint.lane_id
                == npc_waypoint.lane_id
            )
        ]

        if same_lane:

            target_waypoint = (
                same_lane[0]
            )

        else:

            target_waypoint = (
                next_waypoints[0]
            )

        target_transform = (
            target_waypoint.transform
        )

        target_transform.location.z += 0.3

        self.npc_vehicle.set_transform(
            target_transform
        )

        new_distance = (
            self.ego_vehicle
            .get_location()
            .distance(
                self.npc_vehicle.get_location()
            )
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "TEST EVENT: NPC CLEARED"
        )

        print(
            "NPC moved away."
        )

        print(
            f"New true distance: "
            f"{new_distance:.2f} m"
        )

        print(
            "=" * 70
        )

        return True

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
    # DESTROY
    # -------------------------------------------------

    def destroy(self):

        for light in self.test_traffic_light_group.values():
            try:
                light.freeze(False)
            except Exception:
                pass

        self.test_traffic_light = None
        self.test_traffic_light_group = {}
        self.test_traffic_light_state = None
        self._last_traffic_light_verification = None

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
