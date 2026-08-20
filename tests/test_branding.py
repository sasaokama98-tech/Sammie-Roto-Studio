import unittest

from PySide6.QtGui import QImage

from sammie.branding import APP_NAME, SPLASH_IMAGE_PATH


class BrandingTests(unittest.TestCase):
    def test_studio_brand_name(self):
        self.assertEqual(APP_NAME, "Sammie Roto Studio")

    def test_studio_splash_is_packaged_at_expected_size(self):
        image = QImage(str(SPLASH_IMAGE_PATH))

        self.assertFalse(image.isNull())
        self.assertEqual(image.width(), 640)
        self.assertEqual(image.height(), 400)


if __name__ == "__main__":
    unittest.main()
