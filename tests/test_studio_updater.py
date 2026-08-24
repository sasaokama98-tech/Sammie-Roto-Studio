import os
import tempfile
import unittest
from unittest import mock

from dulwich import porcelain
from dulwich.repo import Repo

import manage


class StudioUpdaterTests(unittest.TestCase):
    def test_updater_targets_studio_repository(self):
        self.assertEqual(manage.APP_NAME, "Sammie Roto Studio")
        self.assertEqual(
            manage.REPO_URL,
            "https://github.com/sasaokama98-tech/Sammie-Roto-Studio.git",
        )

    def test_sync_keeps_all_studio_backend_extras(self):
        with mock.patch.object(manage, "run_command") as run_command:
            manage.sync_env("cu130")

        command = run_command.call_args.args[0]
        self.assertIn("--frozen", command)
        for extra in ("cu130", "sam31", "vitmatte", "mematte"):
            self.assertIn(["--extra", extra], [command[i : i + 2] for i in range(len(command) - 1)])

    def test_existing_origin_is_repointed_before_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                repo = Repo.init(".")
                porcelain.remote_add(repo, "origin", "https://example.invalid/upstream.git")

                manage.init_git_tracking()

                updated = Repo(".").get_config().get(
                    (b"remote", b"origin"), b"url"
                )
                self.assertEqual(updated, manage.REPO_URL.encode("utf-8"))
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
