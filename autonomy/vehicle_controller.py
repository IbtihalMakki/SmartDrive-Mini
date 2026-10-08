import carla


class VehicleController:
    def __init__(self, vehicle):
        self.vehicle = vehicle

    def apply_action(self, action, steer=0.0):
        action = action.upper()

        if action == "CONTINUE":
            throttle = 0.35
            brake = 0.0

        elif action == "SLOW_DOWN":
            throttle = 0.12
            brake = 0.0

        elif action == "CAUTION":
            throttle = 0.08
            brake = 0.0

        elif action in ("STOP", "WAIT"):
            throttle = 0.0
            brake = 1.0

        elif action == "RESUME":
            throttle = 0.25
            brake = 0.0

        else:
            throttle = 0.0
            brake = 1.0

        control = carla.VehicleControl(
            throttle=throttle,
            steer=float(steer),
            brake=brake
        )

        self.vehicle.apply_control(control)

        return control