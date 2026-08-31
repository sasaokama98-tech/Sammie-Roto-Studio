import os
import tempfile
import unittest
from pathlib import Path

from sammie.media_input import (
    discover_image_sequences,
    is_supported_drop_path,
    sequence_display_name,
)


class MediaInputTests(unittest.TestCase):
    def test_discovers_direct_numbered_sequence_in_natural_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("shot.1003.jpg", "shot.1001.jpg", "shot.1002.jpg"):
                (root / name).touch()

            sequences = discover_image_sequences(directory)

        self.assertEqual(len(sequences), 1)
        self.assertEqual(
            [os.path.basename(path) for path in sequences[0]],
            ["shot.1001.jpg", "shot.1002.jpg", "shot.1003.jpg"],
        )

    def test_discovers_multiple_sequences_without_combining_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "beauty.1001.png",
                "beauty.1002.png",
                "matte.1001.png",
                "matte.1002.png",
            ):
                (root / name).touch()

            sequences = discover_image_sequences(directory)

        self.assertEqual(len(sequences), 2)
        self.assertTrue(sequence_display_name(sequences[0]).endswith("(2 frames)"))

    def test_folder_search_is_non_recursive_and_ignores_movie_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.mov").touch()
            nested = root / "frames"
            nested.mkdir()
            (nested / "shot.1001.jpg").touch()
            (nested / "shot.1002.jpg").touch()

            sequences = discover_image_sequences(directory)

        self.assertEqual(sequences, [])

    def test_drop_paths_accept_sequence_folders_but_not_movie_folders_as_movies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            movie = root / "clip.mov"
            movie.touch()

            self.assertTrue(is_supported_drop_path(directory))
            self.assertTrue(is_supported_drop_path(str(movie)))
            self.assertFalse(is_supported_drop_path(str(root / "missing.mov")))


if __name__ == "__main__":
    unittest.main()
