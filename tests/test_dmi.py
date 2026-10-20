"""
Unit tests for the DMI (Dry Matter Intake) prediction module, input validation,
SQLite persistence, and Digital Twin integration.
"""
import unittest
import tempfile
from pathlib import Path

from src.config import (
    DMI_FEATURE_NAMES,
    DMI_MODEL_NAME,
)
from src.database import (
    init_db,
    create_animal,
    add_dmi_prediction,
    get_dmi_predictions,
    get_latest_dmi_prediction,
)
from src.digital_twin import (
    get_digital_twin,
    record_dmi_prediction,
)
from src.dmi_model import (
    check_dmi_model_availability,
    validate_dmi_features,
    predict_dmi,
)


class TestDMIModule(unittest.TestCase):
    def setUp(self):
        # Create isolated temporary database for tests
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test_dmi.db"
        init_db(self.db_path)

        # Baseline valid 9-feature set conforming to Pooled Expanded Extra Trees model
        self.valid_features = {
            "age": 48.0,
            "parity": 2.0,
            "DIM": 120.0,
            "BW": 620.0,
            "BCS": 3.0,
            "Milk": 28.5,
            "PC_Milk": 32.5,
            "Fat_Milk": 38.0,
            "Lac_Milk": 48.0,
        }

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_dmi_database_persistence(self):
        """Verify DMI predictions are stored and retrieved from SQLite."""
        create_animal("DMI_TEST_01", species="Cattle", name="Daisy", db_path=self.db_path)

        # Record DMI prediction
        row_id = add_dmi_prediction(
            animal_id="DMI_TEST_01",
            predicted_dmi_kg_day=24.35,
            model_name=DMI_MODEL_NAME,
            input_features=self.valid_features,
            notes="Mid-lactation intake estimate",
            db_path=self.db_path,
        )
        self.assertGreater(row_id, 0)

        # Retrieve history
        history = get_dmi_predictions("DMI_TEST_01", db_path=self.db_path)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["predicted_dmi_kg_day"], 24.35)
        self.assertEqual(history[0]["model_name"], DMI_MODEL_NAME)
        self.assertEqual(history[0]["input_features"]["BW"], 620.0)

        # Retrieve latest
        latest = get_latest_dmi_prediction("DMI_TEST_01", db_path=self.db_path)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["predicted_dmi_kg_day"], 24.35)

    def test_feature_validation_valid(self):
        """Verify valid 9-feature dictionary passes validation."""
        validated, errors = validate_dmi_features(self.valid_features)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(validated)
        self.assertEqual(len(validated), 9)
        for fname in DMI_FEATURE_NAMES:
            self.assertIn(fname, validated)

    def test_feature_validation_missing_feature(self):
        """Verify missing features are flagged with descriptive errors."""
        incomplete = self.valid_features.copy()
        del incomplete["Milk"]

        validated, errors = validate_dmi_features(incomplete)
        self.assertIsNone(validated)
        self.assertTrue(any("Milk" in err for err in errors))

    def test_feature_validation_invalid_types_and_ranges(self):
        """Verify non-numeric values and out-of-range numbers are caught."""
        invalid_features = self.valid_features.copy()
        invalid_features["BW"] = "invalid_string"
        invalid_features["BCS"] = 8.5  # Max is 5.0

        validated, errors = validate_dmi_features(invalid_features)
        self.assertIsNone(validated)
        self.assertTrue(any("BW" in err for err in errors))
        self.assertTrue(any("BCS" in err for err in errors))

    def test_feature_validation_rejects_non_finite_values(self):
        invalid_features = self.valid_features.copy()
        invalid_features["Milk"] = float("nan")
        validated, errors = validate_dmi_features(invalid_features)
        self.assertIsNone(validated)
        self.assertTrue(any("finite" in err for err in errors))

    def test_graceful_missing_model_handling(self):
        """Verify application gracefully reports model artifact requirement when file is missing."""
        dummy_path = Path("models/non_existent_dmi_model.joblib")
        status = check_dmi_model_availability(dummy_path)
        self.assertFalse(status["is_available"])
        self.assertEqual(status["feature_count"], 9)

        # Running predict_dmi without model artifact returns clean status
        result = predict_dmi(self.valid_features, model_path=dummy_path)
        self.assertEqual(result["status"], "model_required")
        self.assertEqual(result["error_type"], "MODEL_ARTIFACT_REQUIRED")
        self.assertIn("validated_inputs", result)

    def test_trained_model_prediction_uses_ordered_features(self):
        result = predict_dmi(self.valid_features)
        self.assertEqual(result["status"], "success")
        self.assertEqual(list(result["input_features"]), DMI_FEATURE_NAMES)
        self.assertEqual(result["unit"], "kg/day")
        self.assertTrue(result["prediction_timestamp"])

    def test_digital_twin_dmi_integration(self):
        """Verify DMI prediction records attach seamlessly to Digital Twin state."""
        create_animal("DMI_TWIN_01", species="Cattle", name="Bella", db_path=self.db_path)

        # Record via digital twin helper
        dt = record_dmi_prediction(
            animal_id="DMI_TWIN_01",
            predicted_dmi_kg_day=22.80,
            model_name=DMI_MODEL_NAME,
            input_features=self.valid_features,
            notes="First DMI record",
            db_path=self.db_path,
        )

        self.assertIsNotNone(dt["latest_dmi"])
        self.assertEqual(dt["latest_dmi"]["predicted_dmi_kg_day"], 22.80)
        self.assertEqual(len(dt["dmi_history"]), 1)


if __name__ == "__main__":
    unittest.main()
