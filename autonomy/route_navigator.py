"""Destination-based routing; emits steering only, never throttle decisions."""
import math
import os
from pathlib import Path
import sys


def load_global_planner():
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ModuleNotFoundError as error:
        if error.name and not error.name.startswith("agents"):
            raise
        root = Path(os.environ.get("CARLA_ROOT", r"D:\CARLA_0.9.16"))
        api = root / "PythonAPI" / "carla"
        if not (api / "agents" / "navigation" / "global_route_planner.py").is_file():
            raise RuntimeError("Set CARLA_ROOT to the installed CARLA directory containing PythonAPI/carla/agents") from error
        sys.path.insert(0, str(api))
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    return GlobalRoutePlanner


class RouteNavigator:
    def __init__(self, vehicle, route, destination, reach_radius=3.0):
        if len(route) < 10:
            raise ValueError("Validation route must contain at least ten waypoints")
        self.vehicle = vehicle
        self.points = [wp.transform.location for wp, _ in route]
        self.endpoint_gap_m = self.points[-1].distance(destination)
        if self.endpoint_gap_m > 4.0:
            raise ValueError("Route endpoint does not connect to destination within 4 m")
        if self.endpoint_gap_m > 0.05:
            self.points.append(destination)
        self.destination = destination
        self.reach_radius = reach_radius
        self.index = 0
        self.reached = False
        self.exhausted_without_arrival = False
        self.route_length_m = sum(a.distance(b) for a, b in zip(self.points, self.points[1:]))
        self.history = []

    @staticmethod
    def trim_to_destination(route, destination, destination_waypoint):
        """GRP may return an exit segment beyond the requested destination."""
        lane = (destination_waypoint.road_id, destination_waypoint.section_id,
                destination_waypoint.lane_id)
        candidates = [i for i, (wp, _) in enumerate(route)
                      if (wp.road_id, wp.section_id, wp.lane_id) == lane
                      and wp.transform.location.distance(destination) <= 4.0]
        if not candidates:
            raise ValueError("Global route does not approach the destination lane within 4 m")
        end = min(candidates, key=lambda i: route[i][0].transform.location.distance(destination))
        return route[:end+1]

    @classmethod
    def plan(
        cls,
        carla_map,
        vehicle,
        destination_index=None,
        min_route_length_m=300.0,
        max_route_length_m=500.0,
        preferred_route_length_m=400.0,
    ):
        """
        Build a destination-based global route.

        When destination_index is None, prefer spawn points whose straight-line
        distance is close to the desired route length, then validate the actual
        GlobalRoutePlanner route length. This avoids selecting the first short
        40-200 m route and produces a more useful autonomous-navigation demo.
        """
        planner = load_global_planner()(carla_map, sampling_resolution=2.0)
        start = vehicle.get_location()
        spawns = carla_map.get_spawn_points()

        if destination_index is not None and not 0 <= destination_index < len(spawns):
            raise ValueError("Destination index outside map spawn points")

        if destination_index is not None:
            choices = [destination_index]
        else:
            # Euclidean distance is only a cheap ranking heuristic.
            # The actual acceptance criterion below uses the GRP route length.
            choices = sorted(
                range(len(spawns)),
                key=lambda i: abs(
                    start.distance(spawns[i].location) - preferred_route_length_m
                ),
            )

        best_candidate = None
        best_delta = float("inf")

        for index in choices:
            destination = spawns[index].location

            # Skip destinations that are obviously too close for the long demo.
            if destination_index is None and start.distance(destination) < 150.0:
                continue

            try:
                route = planner.trace_route(start, destination)
                route = cls.trim_to_destination(
                    route,
                    destination,
                    carla_map.get_waypoint(destination),
                )
            except Exception:
                if destination_index is not None:
                    raise
                continue

            length = sum(
                a[0].transform.location.distance(b[0].transform.location)
                for a, b in zip(route, route[1:])
            )

            if len(route) < 10:
                continue

            # Exact requested validation band.
            if min_route_length_m <= length <= max_route_length_m:
                result = cls(vehicle, route, destination)
                result.destination_index = index
                return result

            # Keep the closest reachable route as diagnostic information only.
            delta = abs(length - preferred_route_length_m)
            if delta < best_delta:
                best_delta = delta
                best_candidate = (index, length)

        if best_candidate is not None:
            index, length = best_candidate
            raise RuntimeError(
                "No reachable validation route in "
                f"{min_route_length_m:.0f}-{max_route_length_m:.0f} m. "
                f"Closest candidate: destination index={index}, "
                f"route length={length:.1f} m."
            )

        raise RuntimeError(
            "No reachable destination could be planned for the long route test."
        )

    def get_steer(self):
        transform = self.vehicle.get_transform()
        location = transform.location
        while self.index < len(self.points) and location.distance(self.points[self.index]) <= self.reach_radius:
            self.index += 1
        remaining = location.distance(self.destination)
        self.reached = self.index >= len(self.points)-1 and remaining <= self.reach_radius
        self.exhausted_without_arrival = self.index >= len(self.points) and not self.reached
        self.history.append(dict(waypoints_reached=self.index, destination_distance_m=remaining,
                                 position=[location.x, location.y, location.z]))
        if self.reached or self.exhausted_without_arrival:
            return 0.0
        # The same proportional yaw controller as WaypointNavigator, directed
        # toward the selected route rather than an arbitrary next branch.
        target = self.points[min(self.index, len(self.points)-1)]
        yaw = math.degrees(math.atan2(target.y-location.y, target.x-location.x))
        error = (yaw-transform.rotation.yaw+180) % 360-180
        return max(-1.0, min(1.0, error/45.0))

    def report(self):
        return dict(status="PASS" if self.reached else "NOT_PASSED",
                    route_generated=True, destination_index=self.destination_index,
                    route_length_m=self.route_length_m, waypoint_count=len(self.points),
                    destination=[self.destination.x, self.destination.y, self.destination.z],
                    route_points=[[p.x, p.y, p.z] for p in self.points],
                    endpoint_gap_before_append_m=self.endpoint_gap_m,
                    exhausted_without_arrival=self.exhausted_without_arrival,
                    waypoints_reached=self.index, destination_radius_m=self.reach_radius,
                    route_progress=self.index/len(self.points), manual_control=False,
                    history=self.history)
