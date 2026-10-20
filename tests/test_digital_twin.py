"""
Unit tests for the Digital Twin module, SQLite persistence, and multimodal risk assessment.
"""
import unittest
import tempfile
import os
from pathlib import Path

from src.database import (
    init_db,
    create_animal,
    get_animal,
    list_animals,
    delete_animal,
    add_observation,
    get_observations,
    add_sensor_reading,
    get_sensor_readings,
)
from src.digital_twin import (
    get_digital_twin,
    multimodal_risk_assessment,
    generate_health_report,
    generate_explainable_summary,
    record_cv_observation,
    record_sensor_reading,
)


class TestDigitalTwinModule(unittest.TestCase):
    def setUp(self):
        # Create a temporary database for each test to ensure test isolation
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test_digital_twins.db"
        init_db(self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_default_seeding(self):
        """Verify default animals (including notebook prototype C001) are seeded."""
        animals = list_animals(self.db_path)
        animal_ids = [a["animal_id"] for a in animals]
        self.assertIn("C001", animal_ids)
        self.assertIn("C002", animal_ids)
        self.assertIn("C003", animal_ids)

        # Check C001 initial state matching notebook
        dt_c001 = get_digital_twin("C001", self.db_path)
        self.assertIsNotNone(dt_c001)
        self.assertEqual(dt_c001["animal_id"], "C001")
        self.assertEqual(dt_c001["current_state"]["predicted_condition"], "healthy")
        self.assertEqual(dt_c001["risk_assessment"]["risk_score"], 6)
        self.assertEqual(dt_c001["risk_assessment"]["status"], "High")

    def test_create_and_manage_animal_profiles(self):
        """Verify unique animal ID creation and duplicate prevention."""
        success = create_animal("COW_100", species="Cattle", name="Daisy-2", breed="Jersey", db_path=self.db_path)
        self.assertTrue(success)

        animal = get_animal("COW_100", self.db_path)
        self.assertEqual(animal["name"], "Daisy-2")
        self.assertEqual(animal["breed"], "Jersey")

        # Duplicate ID should raise ValueError
        with self.assertRaises(ValueError):
            create_animal("COW_100", species="Cattle", db_path=self.db_path)

        # Test deletion
        deleted = delete_animal("COW_100", self.db_path)
        self.assertTrue(deleted)
        self.assertIsNone(get_animal("COW_100", self.db_path))

    def test_record_cv_observation_and_history(self):
        """Verify timestamped CV observations and history accumulation."""
        create_animal("COW_101", species="Cattle", name="Moo", db_path=self.db_path)

        dt = record_cv_observation(
            animal_id="COW_101",
            cv_prediction="foot-and-mouth",
            cv_confidence=0.995,
            model_used="DenseNet-121",
            notes="Lesions present",
            db_path=self.db_path,
        )

        self.assertEqual(dt["current_state"]["predicted_condition"], "foot-and-mouth")
        self.assertEqual(len(dt["history"]), 1)
        self.assertEqual(dt["history"][0]["cv_prediction"], "foot-and-mouth")
        # Current FMD condition contributes 4 points -> Moderate risk
        self.assertEqual(dt["risk_assessment"]["risk_score"], 4)
        self.assertEqual(dt["risk_assessment"]["status"], "Moderate")

    def test_record_sensors_and_provenance(self):
        """Verify sensor telemetry and clear manual vs simulated provenance."""
        create_animal("COW_102", species="Cattle", name="Bessie", db_path=self.db_path)

        # Simulated sensor reading
        dt = record_sensor_reading(
            animal_id="COW_102",
            body_temperature_c=39.4,
            activity_level="Low",
            feed_intake="Reduced",
            ambient_temperature_c=32.0,
            humidity_percent=75.0,
            data_source="Simulated (Collar Sensor)",
            is_simulated=True,
            db_path=self.db_path,
        )

        sensor_data = dt["sensor_data"]
        self.assertTrue(sensor_data["is_simulated"])
        self.assertEqual(sensor_data["body_temperature_c"], 39.4)
        self.assertEqual(sensor_data["activity_level"], "Low")
        self.assertEqual(sensor_data["feed_intake"], "Reduced")

        env_data = dt["environment_data"]
        self.assertEqual(env_data["ambient_temperature_c"], 32.0)
        self.assertEqual(env_data["humidity_percent"], 75.0)

        # Temp >= 39 (+1), Low activity (+1), Reduced feed (+1), Ambient >= 30 (+1), Humidity >= 70 (+1) = 5
        self.assertEqual(dt["risk_assessment"]["risk_score"], 5)
        self.assertEqual(dt["risk_assessment"]["status"], "Moderate")

    def test_multimodal_risk_assessment_scoring_rules(self):
        """
        Verify all individual contributing factors of the scoring rule algorithm:
        - Current disease: +4
        - Previous disease: +1
        - Elevated body temp (>= 39.0°C): +1
        - Low activity: +1
        - Reduced feed: +1
        - High ambient temp (>= 30.0°C): +1
        - High humidity (>= 70.0%): +1
        Total potential max: 10 points.
        """
        create_animal("COW_103", species="Cattle", db_path=self.db_path)

        # Step 1: Baseline healthy observation -> 0 pts (Low)
        record_cv_observation("COW_103", cv_prediction="healthy", cv_confidence=0.99, db_path=self.db_path)
        dt = get_digital_twin("COW_103", self.db_path)
        self.assertEqual(dt["risk_assessment"]["risk_score"], 0)
        self.assertEqual(dt["risk_assessment"]["status"], "Low")

        # Step 2: Past disease observation -> lumpy
        record_cv_observation("COW_103", cv_prediction="lumpy", cv_confidence=0.95, db_path=self.db_path)
        dt = get_digital_twin("COW_103", self.db_path)
        # Current lumpy = 4 pts
        self.assertEqual(dt["risk_assessment"]["risk_score"], 4)
        self.assertEqual(dt["risk_assessment"]["status"], "Moderate")

        # Step 3: Now third observation is healthy, but previous observation has lumpy -> +1 pt
        record_cv_observation("COW_103", cv_prediction="healthy", cv_confidence=0.98, db_path=self.db_path)
        dt = get_digital_twin("COW_103", self.db_path)
        self.assertEqual(dt["risk_assessment"]["risk_score"], 1)

        # Step 4: Add all adverse sensor & environmental conditions -> +5 pts
        # 1 (past disease) + 1 (temp) + 1 (activity) + 1 (feed) + 1 (ambient) + 1 (humidity) = 6 pts -> High
        record_sensor_reading(
            "COW_103",
            body_temperature_c=39.1,
            activity_level="Low",
            feed_intake="Reduced",
            ambient_temperature_c=30.5,
            humidity_percent=72.0,
            db_path=self.db_path,
        )
        dt = get_digital_twin("COW_103", self.db_path)
        self.assertEqual(dt["risk_assessment"]["risk_score"], 6)
        self.assertEqual(dt["risk_assessment"]["status"], "High")

    def test_explainable_summary_generation(self):
        """Verify the 6 explainable report sections are generated and grounded."""
        dt_c001 = get_digital_twin("C001", self.db_path)
        summary = generate_explainable_summary(dt_c001)

        required_sections = [
            "animal_summary",
            "current_observation",
            "historical_trend",
            "contributing_factors",
            "risk_assessment",
            "recommended_follow_up",
        ]
        for sec in required_sections:
            self.assertIn(sec, summary)
            self.assertTrue(len(summary[sec]) > 0)


if __name__ == "__main__":
    unittest.main()
