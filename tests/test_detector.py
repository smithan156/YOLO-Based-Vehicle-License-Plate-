import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.detector import detect_image


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.image = object()
        self.annotated_bgr = object()
        self.annotated_rgb = object()
        self.cv2 = SimpleNamespace(
            IMREAD_COLOR=1,
            COLOR_BGR2RGB=2,
            imdecode=lambda encoded, mode: self.image,
            cvtColor=lambda image, conversion: self.annotated_rgb,
        )
        self.numpy = SimpleNamespace(uint8="uint8", frombuffer=lambda data, dtype: b"encoded")

    def test_returns_annotated_image_and_detection_rows(self):
        box = SimpleNamespace(cls=[1], conf=[0.91], xyxy=[[1.2, 2.4, 30.6, 40.8]])
        result = SimpleNamespace(
            names={1: "car"},
            boxes=[box],
            plot=lambda: self.annotated_bgr,
        )
        calls = []

        def model(image, **kwargs):
            calls.append((image, kwargs))
            return [result]

        with patch.dict(sys.modules, {"cv2": self.cv2, "numpy": self.numpy}):
            annotated, detections = detect_image(b"image bytes", model, 0.4)

        self.assertIs(annotated, self.annotated_rgb)
        self.assertEqual(
            detections,
            [
                {
                    "Detection type": "car",
                    "Detection confidence": 0.91,
                    "x1": 1,
                    "y1": 2,
                    "x2": 31,
                    "y2": 41,
                }
            ],
        )
        self.assertEqual(calls, [(self.image, {"conf": 0.4, "verbose": False})])

    def test_rejects_empty_image_before_loading_image_dependencies(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            detect_image(b"", object())

    def test_rejects_invalid_confidence(self):
        with self.assertRaisesRegex(ValueError, "between"):
            detect_image(b"image bytes", object(), 1.1)

    def test_rejects_image_that_cannot_be_decoded(self):
        cv2 = SimpleNamespace(IMREAD_COLOR=1, imdecode=lambda encoded, mode: None)
        with patch.dict(sys.modules, {"cv2": cv2, "numpy": self.numpy}):
            with self.assertRaisesRegex(ValueError, "could not be decoded"):
                detect_image(b"not an image", object())


if __name__ == "__main__":
    unittest.main()
