from src.feed_shortage import calculate_feed_coverage

result = calculate_feed_coverage(
    inventory=[{
        "feed_name": "Test forage",
        "available_stock_kg": 100,
        "daily_usage_kg": 20,
        "supplier_lead_time_days": 6,
        "reorder_threshold_kg": 50,
    }],
    daily_herd_demand_kg=20,
    planning_days=7,
)

item = result[0]
assert item["days_remaining"] == 5.0
assert item["projected_demand_kg"] == 140.0
assert item["projected_shortage_kg"] == 40.0
assert item["supplier_lead_time_risk"] is True

print("Feed coverage and shortage tests passed.")
