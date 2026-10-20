
from src.procurement_optimizer import optimize_procurement

inventory = [
    {
        "feed_name": "Maize",
        "available_stock_kg": 30,
        "daily_usage_kg": 10,
        "unit_cost": 2,
        "supplier_name": "Prakash",
        "supplier_lead_time_days": 2,
        "reorder_threshold_kg": 20,
    },
    {
        "feed_name": "Wheat",
        "available_stock_kg": 80,
        "daily_usage_kg": 4,
        "unit_cost": 2,
        "supplier_name": "Kamalesh",
        "supplier_lead_time_days": 2,
        "reorder_threshold_kg": 30,
    },
]

result = optimize_procurement(
    inventory,
    planning_horizon_days=7,
    safety_stock_days=3,
    budget=100,
)

print("Total required cost:", result["total_required_cost"])
print("Total allocated cost:", result["total_allocated_cost"])
print("Remaining budget:", result["remaining_budget"])

for item in result["recommendations"]:
    print("\n", item["feed_name"])
    print("Priority:", item["priority"])
    print("Reason:", item["reason"])
    print("Required quantity:", item["required_order_kg"], "kg")
    print("Recommended quantity:", item["recommended_order_kg"], "kg")
    print("Estimated cost:", item["estimated_purchase_cost"])
    print("Order date:", item["suggested_order_date"])
    print("Budget limited:", item["budget_limited"])
