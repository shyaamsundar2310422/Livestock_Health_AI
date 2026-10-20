"""Tests for feed inventory validation, projections, and SQLite persistence."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.database import get_feed_inventory, init_db, upsert_feed_inventory_item
from src.feed_inventory import calculate_inventory_status, validate_feed_inventory_item


class TestFeedInventory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "feed_inventory.db"
        init_db(self.db_path)
        self.item = {
            "feed_name": "Dairy concentrate",
            "available_stock_kg": 40,
            "daily_usage_kg": 10,
            "unit_cost": 1.5,
            "supplier_name": "Local supplier",
            "supplier_lead_time_days": 5,
            "reorder_threshold_kg": 10,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_inventory_calculations_and_lead_time_warning(self):
        result = calculate_inventory_status(
            self.item, as_of=datetime(2026, 1, 1, 12, 0, 0)
        )
        self.assertEqual(result["days_of_stock_remaining"], 4)
        self.assertEqual(result["estimated_stock_out_date"], datetime(2026, 1, 5, 12, 0, 0))
        self.assertFalse(result["reorder_required"])
        self.assertTrue(result["stock_out_before_lead_time"])
        self.assertEqual(result["estimated_inventory_value"], 60)

    def test_zero_usage_has_no_stockout_estimate(self):
        item = {**self.item, "available_stock_kg": 0, "daily_usage_kg": 0}
        result = calculate_inventory_status(item, as_of=datetime(2026, 1, 1))
        self.assertIsNone(result["days_of_stock_remaining"])
        self.assertIsNone(result["estimated_stock_out_date"])
        self.assertFalse(result["stock_out_before_lead_time"])
        self.assertTrue(result["reorder_required"])

    def test_validation_rejects_empty_name_and_negative_or_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "name cannot be empty"):
            validate_feed_inventory_item({**self.item, "feed_name": "  "})
        for field in (
            "available_stock_kg",
            "daily_usage_kg",
            "unit_cost",
            "supplier_lead_time_days",
            "reorder_threshold_kg",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    validate_feed_inventory_item({**self.item, field: -1})
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_feed_inventory_item({**self.item, "unit_cost": float("inf")})

    def test_inventory_starts_empty_and_upsert_persists_updates(self):
        self.assertEqual(get_feed_inventory(self.db_path), [])
        row_id = upsert_feed_inventory_item(**self.item, db_path=self.db_path)
        self.assertGreater(row_id, 0)

        updated = {**self.item, "feed_name": "dairy CONCENTRATE", "available_stock_kg": 25}
        same_row_id = upsert_feed_inventory_item(**updated, db_path=self.db_path)
        persisted = get_feed_inventory(self.db_path)

        self.assertEqual(same_row_id, row_id)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["available_stock_kg"], 25)
        self.assertEqual(persisted[0]["daily_usage_kg"], 10)
        self.assertTrue(persisted[0]["last_updated"])


if __name__ == "__main__":
    unittest.main()
