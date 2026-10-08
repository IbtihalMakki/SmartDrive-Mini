import math


class WaypointNavigator:
    def __init__(self, carla_map, vehicle):
        self.map = carla_map
        self.vehicle = vehicle

    def get_steer(self, lookahead=5.0):
        transform = self.vehicle.get_transform()
        location = transform.location

        current_wp = self.map.get_waypoint(
            location,
            project_to_road=True
        )

        next_wps = current_wp.next(lookahead)

        if not next_wps:
            return 0.0

        target_wp = next_wps[0]
        target = target_wp.transform.location

        dx = target.x - location.x
        dy = target.y - location.y

        target_yaw = math.degrees(math.atan2(dy, dx))
        vehicle_yaw = transform.rotation.yaw

        yaw_error = target_yaw - vehicle_yaw

        # Normalize [-180, 180]
        yaw_error = (yaw_error + 180.0) % 360.0 - 180.0

        # Simple proportional steering controller
        steer = yaw_error / 45.0

        return max(-1.0, min(1.0, steer))