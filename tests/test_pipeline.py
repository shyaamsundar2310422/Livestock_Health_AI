"""
Unit tests for the cattle disease prediction pipeline and Grad-CAM explainability.
"""
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

from src.config import (
    CLASS_KEYS,
    CLASS_DISPLAY_NAMES,
    DENSENET121_WEIGHTS,
    EFFICIENTNET_WEIGHTS,
    SAMPLE_IMAGES_DIR,
)
from src.model import (
    load_model,
    predict,
    preprocess_image,
    validate_and_load_image,
)
from src.gradcam import generate_gradcam_visualizations


class TestCattleDiseasePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_healthy = SAMPLE_IMAGES_DIR / "healthy_sample2.jpg"
        cls.sample_fmd = SAMPLE_IMAGES_DIR / "foot_and_mouth_sample1.jpg"
        cls.sample_lumpy = SAMPLE_IMAGES_DIR / "lumpy_skin_sample1.jpg"

    def test_weights_exist(self):
        """Verify that required model weights files exist locally."""
        self.assertTrue(DENSENET121_WEIGHTS.exists(), f"Missing {DENSENET121_WEIGHTS}")
        self.assertTrue(EFFICIENTNET_WEIGHTS.exists(), f"Missing {EFFICIENTNET_WEIGHTS}")

    def test_model_loading_densenet(self):
        """Verify DenseNet-121 loads successfully."""
        model = load_model("densenet121")
        self.assertIsNotNone(model)
        self.assertEqual(model.classifier.out_features, len(CLASS_KEYS))

    def test_model_loading_efficientnet(self):
        """Verify EfficientNet-B0 loads successfully."""
        model = load_model("efficientnet_b0")
        self.assertIsNotNone(model)
        self.assertEqual(model.classifier[1].out_features, len(CLASS_KEYS))

    def test_image_preprocessing(self):
        """Verify image tensor shape is (1, 3, 224, 224)."""
        img = Image.new("RGB", (300, 400), color=(100, 150, 200))
        tensor = preprocess_image(img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_prediction_output_structure(self):
        """Verify prediction dictionary structure and probabilities sum to 1."""
        if not self.sample_healthy.exists():
            self.skipTest("Sample image not found")

        img = validate_and_load_image(self.sample_healthy)
        res = predict(img, model_name="densenet121")

        self.assertIn("predicted_key", res)
        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)

        self.assertIn(res["predicted_key"], CLASS_KEYS)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_prediction_accuracy_on_known_samples(self):
        """Verify DenseNet-121 correctly identifies clear known class samples."""
        if not (self.sample_healthy.exists() and self.sample_fmd.exists() and self.sample_lumpy.exists()):
            self.skipTest("Sample images not available")

        # Test Healthy
        img_h = validate_and_load_image(self.sample_healthy)
        res_h = predict(img_h, model_name="densenet121")
        self.assertEqual(res_h["predicted_key"], "healthy")

        # Test FMD
        img_fmd = validate_and_load_image(self.sample_fmd)
        res_fmd = predict(img_fmd, model_name="densenet121")
        self.assertEqual(res_fmd["predicted_key"], "foot-and-mouth")

        # Test Lumpy
        img_lumpy = validate_and_load_image(self.sample_lumpy)
        res_lumpy = predict(img_lumpy, model_name="densenet121")
        self.assertEqual(res_lumpy["predicted_key"], "lumpy")

    def test_gradcam_generation(self):
        """Verify Grad-CAM produces heatmap and overlay with matching image dimensions."""
        if not self.sample_healthy.exists():
            self.skipTest("Sample image not found")

        img = validate_and_load_image(self.sample_healthy)
        for model_name in ["densenet121", "efficientnet_b0"]:
            with self.subTest(model=model_name):
                cam_res = generate_gradcam_visualizations(img, model_name=model_name, alpha=0.5)
                self.assertIn("heatmap", cam_res)
                self.assertIn("overlay", cam_res)
                self.assertIn("cam", cam_res)

                # Check dimensions match original image
                self.assertEqual(cam_res["overlay"].size, img.size)
                self.assertEqual(cam_res["heatmap"].size, img.size)

    def test_invalid_image_handling(self):
        """Verify that invalid inputs raise ValueError properly."""
        with self.assertRaises(ValueError):
            validate_and_load_image(Path("non_existent_file.jpg"))

        tiny_img = Image.new("RGB", (4, 4))
        with self.assertRaises(ValueError):
            validate_and_load_image(tiny_img)


if __name__ == "__main__":
    unittest.main()
