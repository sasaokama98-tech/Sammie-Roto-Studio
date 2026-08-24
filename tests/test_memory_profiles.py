import unittest
from unittest.mock import patch
import tempfile

from sammie.memory_profiles import (
    BALANCED,
    CUSTOM,
    FAST,
    MEMORY_SAFE,
    apply_memory_profile,
    get_memory_profile,
)
from sammie.settings_manager import SettingsManager


class _Settings:
    def __init__(self):
        self.values = {}

    def set_session_setting(self, key, value):
        self.values[key] = value
        return True


class MemoryProfileTests(unittest.TestCase):
    def test_profiles_scale_resolution_tile_and_token_budget(self):
        safe = get_memory_profile(MEMORY_SAFE).settings
        balanced = get_memory_profile(BALANCED).settings
        fast = get_memory_profile(FAST).settings

        self.assertLess(safe["matany_res"], balanced["matany_res"])
        self.assertLess(balanced["matany_res"], fast["matany_res"])
        self.assertLess(safe["mematte_tile_size"], balanced["mematte_tile_size"])
        self.assertLess(
            balanced["mematte_max_tokens"], fast["mematte_max_tokens"]
        )
        self.assertEqual(safe["mematte_precision"], "Float16")

    def test_apply_updates_session_and_custom_preserves_values(self):
        settings = _Settings()
        values = apply_memory_profile(settings, MEMORY_SAFE)
        self.assertEqual(settings.values["memory_profile"], MEMORY_SAFE)
        self.assertEqual(settings.values["mematte_tile_size"], 1024)
        self.assertEqual(values["hybrid_flow_resolution"], 480)

        before = dict(settings.values)
        self.assertEqual(apply_memory_profile(settings, CUSTOM), {})
        self.assertEqual(settings.values["memory_profile"], CUSTOM)
        for key, value in before.items():
            if key != "memory_profile":
                self.assertEqual(settings.values[key], value)

    def test_new_session_applies_configured_default_profile(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "sammie.settings_manager.os.path.exists", return_value=False
        ):
            manager = SettingsManager(temp_dir=directory)
        manager.app_settings.default_memory_profile = MEMORY_SAFE
        manager.create_new_session("fixture.mov")

        self.assertEqual(manager.session_settings.memory_profile, MEMORY_SAFE)
        self.assertEqual(manager.session_settings.matany_res, 480)
        self.assertEqual(manager.session_settings.mematte_tile_size, 1024)
        self.assertEqual(manager.session_settings.hybrid_flow_resolution, 480)


if __name__ == "__main__":
    unittest.main()
