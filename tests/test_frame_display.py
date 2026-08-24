import unittest

from sammie.frame_display import (
    FRAME_INDEX_MODE,
    SOURCE_FRAME_MODE,
    TIMECODE_MODE,
    available_frame_display_modes,
    extract_source_frame_number,
    format_frame_value,
    format_in_out_range,
    infer_contiguous_sequence_metadata,
    sequence_frame_metadata,
)


class FrameDisplayTests(unittest.TestCase):
    def test_extracts_final_sequence_token_and_padding(self):
        self.assertEqual(
            extract_source_frame_number("shot.comp.v003.0998.jpg"), (998, 4)
        )
        self.assertIsNone(extract_source_frame_number("shot.comp.jpg"))

    def test_sequence_metadata_preserves_non_contiguous_source_numbers(self):
        numbers, padding = sequence_frame_metadata(
            ["shot.0998.exr", "shot.0999.exr", "shot.1001.exr"]
        )
        self.assertEqual(numbers, [998, 999, 1001])
        self.assertEqual(padding, 4)

    def test_legacy_sequence_metadata_can_be_rebuilt_from_selected_frame(self):
        numbers, padding = infer_contiguous_sequence_metadata(
            "shot.0998.jpg", 4
        )
        self.assertEqual(numbers, [998, 999, 1000, 1001])
        self.assertEqual(padding, 4)

    def test_source_mode_requires_complete_sequence_mapping(self):
        self.assertEqual(
            available_frame_display_modes(3, [998, 999, 1000]),
            [FRAME_INDEX_MODE, SOURCE_FRAME_MODE],
        )
        self.assertEqual(
            available_frame_display_modes(3, [998, 999]), [FRAME_INDEX_MODE]
        )

    def test_timecode_mode_is_ready_but_only_exposed_with_complete_metadata(self):
        modes = available_frame_display_modes(
            2,
            [1001, 1002],
            ["01:00:00:00", "01:00:00:01"],
        )
        self.assertEqual(
            modes, [FRAME_INDEX_MODE, SOURCE_FRAME_MODE, TIMECODE_MODE]
        )

    def test_formats_source_frame_and_in_out_range(self):
        source_numbers = [998, 999, 1000, 1001]
        formatter = lambda index: format_frame_value(
            index,
            SOURCE_FRAME_MODE,
            source_frame_numbers=source_numbers,
            source_frame_padding=4,
        )
        self.assertEqual(formatter(0), "0998")
        self.assertEqual(
            format_in_out_range(1, 3, formatter),
            "In: 0999   Out: 1001   Range: 3 frames",
        )


if __name__ == "__main__":
    unittest.main()
