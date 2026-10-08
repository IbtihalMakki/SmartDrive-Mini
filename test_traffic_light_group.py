"""Regression tests for group forcing, without a running CARLA server."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class TrafficLightGroupTests(unittest.TestCase):
    def setUp(self):
        fake_carla = types.SimpleNamespace(
            Client=Mock(),
            TrafficLightState=types.SimpleNamespace(
                Red="RED", Yellow="YELLOW", Green="GREEN"
            )
        )
        spec = importlib.util.spec_from_file_location(
            "client_under_test",
            Path(__file__).parent / "simulation" / "carla_client.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"carla": fake_carla}):
            spec.loader.exec_module(module)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.client = module.CarlaClient()
        self.selected = Mock(id=10)
        self.affecting = Mock(id=20)
        self.selected.get_group_traffic_lights.return_value = [
            self.selected, self.affecting, self.affecting
        ]
        self.client.test_traffic_light = self.selected
        self.affecting.get_group_traffic_lights.return_value = []
        self.client.ego_vehicle = Mock()
        self.client.ego_vehicle.get_traffic_light.return_value = self.affecting

    def test_other_group_actor_receives_every_phase(self):
        for state in ("RED", "YELLOW", "GREEN"):
            self.client.set_test_traffic_light_state(state.lower())
            for light in (self.selected, self.affecting):
                light.set_state.assert_called_with(state)
                light.freeze.assert_called_with(True)
            self.assertTrue(self.client.verify_test_traffic_light_state(
                {"actor_id": 20, "state": state, "detected": True}
            ))
        self.assertEqual(self.affecting.set_state.call_count, 3)

    def test_stale_missing_and_unrelated_actors_do_not_pass(self):
        self.client.set_test_traffic_light_state("YELLOW")
        for actor_id, state in ((20, "RED"), (None, "NONE"), (30, "YELLOW")):
            self.assertFalse(self.client.verify_test_traffic_light_state(
                {"actor_id": actor_id, "state": state, "detected": actor_id is not None}
            ))
        self.assertTrue(self.client.verify_test_traffic_light_state(
            {"actor_id": 20, "state": "YELLOW", "detected": True}
        ))

    def test_selected_actor_included_when_group_list_omits_it(self):
        self.selected.get_group_traffic_lights.return_value = [self.affecting]
        self.client.set_test_traffic_light_state("GREEN")
        self.selected.set_state.assert_called_once_with("GREEN")

    def test_cleanup_unfreezes_all_group_members(self):
        self.client.set_test_traffic_light_state("RED")
        self.client.destroy()
        for light in (self.selected, self.affecting):
            light.freeze.assert_called_with(False)
        self.assertEqual(self.client.test_traffic_light_group, {})

    def test_freeze_failure_is_not_reported_as_success(self):
        self.affecting.freeze.side_effect = RuntimeError("RPC failed")
        with self.assertRaises(RuntimeError):
            self.client.set_test_traffic_light_state("YELLOW")
        self.assertIsNone(self.client.test_traffic_light_state)
        self.selected.set_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
