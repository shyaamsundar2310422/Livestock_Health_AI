from typing import Any, Dict, List


def calculate_feed_coverage(
    inventory: List[Dict[str, Any]],
    daily_herd_demand_kg: float,
    planning_days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Estimate inventory coverage and shortage for a single feed.

    daily_herd_demand_kg must represent demand for this specific feed,
    not total herd DMI unless the entire ration is this feed.
    """
    if daily_herd_demand_kg < 0:
        raise ValueError("Daily demand cannot be negative")
    if planning_days < 1:
        raise ValueError("Planning days must be at least 1")

    results = []

    for item in inventory:
        name = str(item.get("feed_name", "")).strip()
        if not name:
            continue

        stock = float(item.get("available_stock_kg", 0))
        usage = float(item.get("daily_usage_kg", 0))
        lead_time = float(item.get("supplier_lead_time_days", 0))
        threshold = float(item.get("reorder_threshold_kg", 0))

        if min(stock, usage, lead_time, threshold) < 0:
            raise ValueError(f"Negative inventory value for {name}")

        effective_daily_demand = (
            daily_herd_demand_kg if daily_herd_demand_kg > 0 else usage
        )

        days_remaining = (
            stock / effective_daily_demand
            if effective_daily_demand > 0
            else None
        )
        planned_demand = effective_daily_demand * planning_days
        projected_shortage = max(0.0, planned_demand - stock)

        reorder = stock <= threshold
        lead_time_risk = (
            days_remaining is not None and days_remaining <= lead_time
        )

        results.append({
            "feed_name": name,
            "available_stock_kg": round(stock, 2),
            "daily_demand_kg": round(effective_daily_demand, 2),
            "days_remaining": (
                round(days_remaining, 2)
                if days_remaining is not None else None
            ),
            "planning_days": planning_days,
            "projected_demand_kg": round(planned_demand, 2),
            "projected_shortage_kg": round(projected_shortage, 2),
            "reorder_required": reorder,
            "supplier_lead_time_risk": lead_time_risk,
        })

    return results
