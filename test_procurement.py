from src.feed_inventory import calculate_procurement_recommendation

item = {
    "feed_name": "Maize",
    "available_stock_kg": 30,
    "daily_usage_kg": 1,
    "unit_cost": 2,
    "supplier_name": "Prakash",
    "supplier_lead_time_days": 2,
    "reorder_threshold_kg": 20,
}

result = calculate_procurement_recommendation(
    item,
    planning_horizon_days=7,
    safety_stock_days=3,
)

for key, value in result.items():
    print(f"{key}: {value}")
