import math
import unittest
from types import SimpleNamespace
from autonomy.route_navigator import RouteNavigator


class Point:
    def __init__(self, x, y=0, z=0):
        self.x, self.y, self.z = x, y, z

    def distance(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.pose = SimpleNamespace(location=Point(0), rotation=SimpleNamespace(yaw=0))
        self.vehicle = SimpleNamespace(get_transform=lambda: self.pose)
        route = [(SimpleNamespace(transform=SimpleNamespace(location=Point(i*5))), None) for i in range(11)]
        self.nav = RouteNavigator(self.vehicle, route, Point(50))
        self.nav.destination_index = 3

    def test_ordered_progress_and_arrival(self):
        for i in range(10):
            self.pose.location = Point(i*5)
            self.assertAlmostEqual(self.nav.get_steer(), 0)
            self.assertFalse(self.nav.reached)
        self.pose.location = Point(50)
        self.nav.get_steer()
        self.assertTrue(self.nav.reached)
        self.assertEqual(self.nav.report()["status"], "PASS")
        self.assertEqual(self.nav.report()["waypoints_reached"], 11)

    def test_teleport_to_destination_does_not_pass(self):
        self.pose.location = Point(50)
        self.nav.get_steer()
        self.assertFalse(self.nav.reached)
        self.assertEqual(self.nav.index, 0)

    def test_heading_changes_steering_only(self):
        self.pose.rotation.yaw = 90
        self.assertEqual(self.nav.get_steer(), -1)
        self.assertFalse(hasattr(self.vehicle, "apply_control"))

    def test_route_too_short_is_rejected(self):
        with self.assertRaises(ValueError):
            RouteNavigator(self.vehicle, [], Point(50))

    def test_planner_tail_beyond_destination_is_trimmed(self):
        route = [(SimpleNamespace(road_id=1, section_id=0, lane_id=1,
                                  transform=SimpleNamespace(location=Point(i*2))), None)
                 for i in range(32)]
        destination = Point(41)
        route = RouteNavigator.trim_to_destination(route, destination, route[0][0])
        self.assertEqual(route[-1][0].transform.location.x, 40)
        nav = RouteNavigator(self.vehicle, route, destination)
        self.assertEqual(nav.points[-1].x, 41)
        for x in range(0, 42, 2):
            self.pose.location = Point(x)
            nav.get_steer()
        self.assertTrue(nav.reached)

    def test_wrong_lane_endpoint_is_rejected(self):
        wp = SimpleNamespace(road_id=1, section_id=0, lane_id=1,
                             transform=SimpleNamespace(location=Point(40)))
        target = SimpleNamespace(road_id=1, section_id=0, lane_id=2)
        with self.assertRaises(ValueError):
            RouteNavigator.trim_to_destination([(wp, None)], Point(40), target)

    def test_exhausted_route_never_steers_back_to_old_endpoint(self):
        self.nav.index = len(self.nav.points)
        self.pose.location = Point(62)
        self.assertEqual(self.nav.get_steer(), 0)
        self.assertTrue(self.nav.exhausted_without_arrival)
        self.assertFalse(self.nav.reached)


if __name__ == "__main__":
    unittest.main()
