import os
import stat
import tempfile
import unittest
from pathlib import Path

from sammie import core


class TempWorkspaceTests(unittest.TestCase):
    def test_remove_tree_handles_readonly_git_pack_files(self):
        with tempfile.TemporaryDirectory() as parent:
            tree = Path(parent) / "temp"
            pack_dir = tree / "source_audit" / ".git" / "objects" / "pack"
            pack_dir.mkdir(parents=True)
            pack_file = pack_dir / "fixture.idx"
            pack_file.write_bytes(b"fixture")
            os.chmod(pack_file, stat.S_IREAD)

            core.remove_tree(tree)

            self.assertFalse(tree.exists())


if __name__ == "__main__":
    unittest.main()
