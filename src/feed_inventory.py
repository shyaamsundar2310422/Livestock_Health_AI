"""Validation and inventory projections for the feed supply-chain tab."""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, Optional


NUMERIC_FIELDS = (
    "available_stock_kg",
    "daily_usage_kg",
    "unit_cost",
    "supplier_lead_time_days",
    "reorder_threshold_kg",
)


def validate_feed_inventory_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an inventory item and reject empty names or invalid values."""
    raw_name = item.get("feed_name", "")
    name = raw_name.strip() if isinstance(raw_name, str) else ""
    if not name:
        raise ValueError("Feed/ingredient name cannot be empty.")

    validated: Dict[str, Any] = {"feed_name": name}
    for field in NUMERIC_FIELDS:
        try:
            value = float(item[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a valid non-negative number.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be a finite, non-negative number.")
        validated[field] = value

    validated["supplier_name"] = str(item.get("supplier_name") or "").strip()
    return validated


def calculate_inventory_status(
    item: Dict[str, Any], as_of: Optional[datetime] = None
) -> Dict[str, Any]:
    """Calculate stock coverage, stock-out date, reorder and lead-time warnings."""
    item = validate_feed_inventory_item(item)
    stock = item["available_stock_kg"]
    usage = item["daily_usage_kg"]
    lead_time = item["supplier_lead_time_days"]
    threshold = item["reorder_threshold_kg"]

    days_remaining = stock / usage if usage > 0 else None
    stock_out_date = None
    if days_remaining is not None:
        stock_out_date = (as_of or datetime.now()) + timedelta(days=days_remaining)

    return {
        "days_of_stock_remaining": days_remaining,
        "estimated_stock_out_date": stock_out_date,
        "reorder_required": stock <= threshold,
        "stock_out_before_lead_time": (
            days_remaining is not None and days_remaining < lead_time
        ),
        "estimated_inventory_value": stock * item["unit_cost"],
    }


def calculate_procurement_recommendation(
    item: Dict[str, Any],
    planning_horizon_days: int = 7,
    safety_stock_days: float = 3.0,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Recommend a feed reorder quantity and estimate urgency and cost.

    Quantities use the same units as the inventory item. For the
    current application, these should be kg of dry matter (DM).
    """
    item = validate_feed_inventory_item(item)

    if planning_horizon_days < 1:
        raise ValueError("planning_horizon_days must be at least 1.")
    if not math.isfinite(float(safety_stock_days)) or safety_stock_days < 0:
        raise ValueError("safety_stock_days must be a finite, non-negative number.")

    stock = item["available_stock_kg"]
    usage = item["daily_usage_kg"]
    lead_time = item["supplier_lead_time_days"]
    threshold = item["reorder_threshold_kg"]
    unit_cost = item["unit_cost"]

    # Stock needed to cover the planning horizon plus safety stock.
    target_stock = usage * (planning_horizon_days + safety_stock_days)

    # Never recommend a negative order quantity.
    reorder_quantity = max(0.0, target_stock - stock)
    estimated_cost = reorder_quantity * unit_cost

    days_remaining = stock / usage if usage > 0 else None
    stock_out_date = None
    if days_remaining is not None:
        stock_out_date = (as_of or datetime.now()) + timedelta(
            days=days_remaining
        )

    if usage == 0:
        urgency = "NO USAGE"
        suggested_order_date = None
    elif days_remaining is not None and days_remaining <= lead_time:
        urgency = "URGENT"
        suggested_order_date = as_of or datetime.now()
    elif stock <= threshold or reorder_quantity > 0:
        urgency = "REORDER"
        suggested_order_date = (
            (as_of or datetime.now())
            + timedelta(days=max(0, days_remaining - lead_time))
        )
    else:
        urgency = "STOCK SUFFICIENT"
        suggested_order_date = None

    return {
        "feed_name": item["feed_name"],
        "available_stock_kg": round(stock, 2),
        "daily_usage_kg": round(usage, 2),
        "planning_horizon_days": planning_horizon_days,
        "safety_stock_days": float(safety_stock_days),
        "target_stock_kg": round(target_stock, 2),
        "recommended_order_kg": round(reorder_quantity, 2),
        "unit_cost": round(unit_cost, 2),
        "estimated_purchase_cost": round(estimated_cost, 2),
        "days_of_stock_remaining": (
            round(days_remaining, 2) if days_remaining is not None else None
        ),
        "estimated_stock_out_date": (
            stock_out_date.strftime("%Y-%m-%d") if stock_out_date else None
        ),
        "supplier_name": item["supplier_name"],
        "supplier_lead_time_days": lead_time,
        "urgency": urgency,
        "suggested_order_date": (
            suggested_order_date.strftime("%Y-%m-%d")
            if suggested_order_date else None
        ),
    }
