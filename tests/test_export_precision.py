import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import Imath
import numpy as np
import OpenEXR

from sammie import core
from sammie.export_workers import SequenceExportWorker


class ExportPrecisionTests(unittest.TestCase):
    def test_16_bit_intermediate_survives_float_exr_roundtrip(self):
        values16 = np.array([[0, 1, 32768, 65535]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            matte_dir = os.path.join(directory, "matting")
            frame_dir = os.path.join(matte_dir, "00000")
            os.makedirs(frame_dir)
            self.assertTrue(cv2.imwrite(os.path.join(frame_dir, "0.png"), values16))

            with patch.object(core, "matting_dir", matte_dir):
                loaded = core.load_matte_for_export(0, 0)
            self.assertTrue(np.allclose(loaded, values16 / 65535.0, atol=1e-7))

            exr_path = os.path.join(directory, "precision.exr")
            SequenceExportWorker._write_exr_file(exr_path, {"Object_0.Y": loaded})
            exr = OpenEXR.InputFile(exr_path)
            raw = exr.channel("Object_0.Y", Imath.PixelType(Imath.PixelType.FLOAT))
            restored = np.frombuffer(raw, dtype=np.float32).reshape(values16.shape)
            exr.close()
            self.assertTrue(np.array_equal(restored, loaded))


if __name__ == "__main__":
    unittest.main()
