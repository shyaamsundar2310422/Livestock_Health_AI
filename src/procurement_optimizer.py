
from datetime import date, timedelta
import math

from src.feed_inventory import validate_feed_inventory_item


def validate_supplier_offer(offer):
    """Validate and normalize one manually entered supplier offer."""
    feed_name = str(offer.get("feed_name", "")).strip()
    supplier_name = str(offer.get("supplier_name", "")).strip()
    if not feed_name:
        raise ValueError("Supplier offer ingredient name cannot be empty.")
    if not supplier_name:
        raise ValueError("Supplier offer supplier name cannot be empty.")

    values = {}
    for field in ("unit_cost", "delivery_lead_time_days"):
        try:
            value = float(offer[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Supplier offer {field} must be non-negative.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Supplier offer {field} must be finite and non-negative.")
        values[field] = value

    return {
        "feed_name": feed_name,
        "supplier_name": supplier_name,
        **values,
    }


def parse_supplier_offers(text):
    """Parse manual offers formatted as ingredient|supplier|cost|lead days."""
    offers = []
    for line_number, line in enumerate(str(text or "").splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4:
            raise ValueError(
                f"Supplier offer line {line_number} must contain ingredient, "
                "supplier, unit cost, and delivery lead time separated by |."
            )
        offers.append(validate_supplier_offer({
            "feed_name": fields[0],
            "supplier_name": fields[1],
            "unit_cost": fields[2],
            "delivery_lead_time_days": fields[3],
        }))
    return offers


def select_best_supplier_offer(offers, delivery_deadline_days):
    """Return the lowest-cost offer arriving within stock coverage, or None.

    The delivery deadline is the ingredient's remaining stock coverage in days
    (available stock / planned daily demand), matching procurement's existing
    stock-out and lead-time urgency logic. Ties resolve by lead time, then
    supplier name, so repeated runs return the same offer.
    """
    if delivery_deadline_days is None:
        return None
    try:
        deadline = float(delivery_deadline_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("Delivery deadline must be a finite, non-negative number.") from exc
    if not math.isfinite(deadline) or deadline < 0:
        raise ValueError("Delivery deadline must be a finite, non-negative number.")

    eligible = [
        validate_supplier_offer(offer)
        for offer in offers
    ]
    eligible = [offer for offer in eligible if offer["delivery_lead_time_days"] <= deadline]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda offer: (
            offer["unit_cost"],
            offer["delivery_lead_time_days"],
            offer["supplier_name"].casefold(),
            offer["supplier_name"],
        ),
    )


def resolve_saved_herd_dmi(animals, prediction_lookup):
    """Sum each registered animal's latest valid saved DMI prediction.

    A missing, unreadable, non-finite, zero, or negative latest prediction is
    reported as an issue. Older records are not substituted for an invalid
    latest record.
    """
    totals = {}
    issues = []

    for animal in animals:
        animal_id = str(animal.get("animal_id", "")).strip()
        if not animal_id:
            issues.append("Registered animal has no animal ID")
            continue
        try:
            prediction = prediction_lookup(animal_id)
        except Exception as exc:
            issues.append(f"{animal_id}: could not load latest prediction ({exc})")
            continue

        if not prediction:
            issues.append(f"{animal_id}: no saved DMI prediction")
            continue
        try:
            value = float(prediction.get("predicted_dmi_kg_day"))
        except (AttributeError, TypeError, ValueError):
            issues.append(f"{animal_id}: latest DMI prediction is not numeric")
            continue
        if not math.isfinite(value) or value <= 0:
            issues.append(f"{animal_id}: latest DMI prediction must be finite and greater than zero")
            continue
        totals[animal_id] = value

    total_dmi = sum(totals.values())
    if not math.isfinite(total_dmi):
        issues.append("Total herd DMI is not finite")
    if not animals:
        issues.append("No registered animals are available")

    return {
        "animal_dmi_kg_day": totals,
        "total_dmi_kg_day": total_dmi,
        "issues": issues,
    }


def calculate_ration_demand(
    inventory,
    herd_daily_dmi_kg,
    ration_percentages,
    planning_horizon_days,
):
    """Calculate ingredient daily/horizon demand from a 100% ration mix."""
    herd_daily_dmi_kg = float(herd_daily_dmi_kg)
    if not math.isfinite(herd_daily_dmi_kg) or herd_daily_dmi_kg <= 0:
        raise ValueError("Herd daily DMI must be finite and greater than zero.")
    if int(planning_horizon_days) < 1:
        raise ValueError("Planning horizon must be at least 1 day.")
    planning_horizon_days = int(planning_horizon_days)

    inventory = list(inventory)
    names = [str(item.get("feed_name", "")).strip() for item in inventory]
    if not inventory or any(not name for name in names):
        raise ValueError("At least one named inventory ingredient is required.")
    if len(set(name.lower() for name in names)) != len(names):
        raise ValueError("Inventory ingredient names must be unique.")

    shares = {}
    for name in names:
        try:
            share = float(ration_percentages[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"A ration percentage is required for {name}.") from exc
        if not math.isfinite(share) or share < 0 or share > 100:
            raise ValueError(f"Ration percentage for {name} must be between 0 and 100.")
        shares[name] = share

    if not 99.9 <= sum(shares.values()) <= 100.1:
        raise ValueError("Ration percentages must total 100%.")

    rows = []
    for raw_item, name in zip(inventory, names):
        item = validate_feed_inventory_item(raw_item)
        share_pct = shares[name]
        daily_demand = herd_daily_dmi_kg * (share_pct / 100.0)
        horizon_demand = daily_demand * planning_horizon_days
        stock = item["available_stock_kg"]
        days_remaining = stock / daily_demand if daily_demand > 0 else None
        rows.append({
            **item,
            "ration_share_pct": share_pct,
            "daily_demand_kg": daily_demand,
            "horizon_demand_kg": horizon_demand,
            "projected_shortage_kg": max(0.0, horizon_demand - stock),
            "days_of_stock_remaining": days_remaining,
            "reorder_required": (
                stock <= item["reorder_threshold_kg"]
                or horizon_demand > stock
            ),
            "lead_time_risk": (
                days_remaining is not None
                and days_remaining < item["supplier_lead_time_days"]
            ),
        })
    return rows


def optimize_procurement(
    inventory,
    planning_horizon_days=7,
    safety_stock_days=3,
    budget=None,
):
    """
    Rank feed purchase recommendations using projected demand,
    safety stock, supplier lead time, and an optional budget.

    Inventory quantities must use the same units (kg DM if the
    stored inventory represents dry matter).
    """

    if planning_horizon_days < 1:
        raise ValueError("Planning horizon must be at least 1 day.")

    if safety_stock_days < 0:
        raise ValueError("Safety stock days cannot be negative.")

    if budget is not None:
        budget = float(budget)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("Budget must be finite and non-negative.")

    recommendations = []

    for item in inventory:
        name = str(item.get("feed_name", "")).strip()
        if not name:
            continue

        stock = float(item.get("available_stock_kg", 0))
        usage = float(item.get("daily_usage_kg", 0))
        cost = float(item.get("unit_cost", 0))
        lead = float(item.get("supplier_lead_time_days", 0))
        reorder_level = float(item.get("reorder_threshold_kg", 0))

        values = [stock, usage, cost, lead, reorder_level]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError(
                f"Invalid negative or non-finite value for {name}."
            )

        days_left = stock / usage if usage > 0 else math.inf

        # Order enough for the planning horizon plus safety stock.
        target_stock = usage * (
            planning_horizon_days + safety_stock_days
        )

        # Respect a manually configured reorder threshold as well.
        target_stock = max(target_stock, reorder_level)

        required_order = max(0.0, target_stock - stock)
        required_order = math.ceil(required_order * 100) / 100

        lead_time_risk = (
            usage > 0 and days_left <= lead + safety_stock_days
        )
        reorder_needed = (
            required_order > 0
            or stock <= reorder_level
        )

        if lead_time_risk:
            priority = "CRITICAL"
            reason = (
                f"Stock covers about {days_left:.1f} days, "
                f"but supplier lead time is {lead:.1f} days."
            )
        elif reorder_needed:
            priority = "HIGH" if days_left <= lead + safety_stock_days else "MEDIUM"
            reason = (
                f"Target stock is {target_stock:.2f} kg; "
                f"current stock is {stock:.2f} kg."
            )
        else:
            priority = "LOW"
            reason = "Current stock covers the target level."

        recommendations.append({
            "feed_name": name,
            "available_stock_kg": round(stock, 2),
            "daily_usage_kg": round(usage, 2),
            "days_of_stock_remaining": (
                round(days_left, 1) if math.isfinite(days_left)
                else None
            ),
            "supplier_lead_time_days": round(lead, 1),
            "target_stock_kg": round(target_stock, 2),
            "required_order_kg": round(required_order, 2),
            "unit_cost": round(cost, 2),
            "required_cost": round(required_order * cost, 2),
            "priority": priority,
            "reason": reason,
            "supplier_name": item.get("supplier_name", ""),
        })

    # Resolve ties deterministically.
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(
        key=lambda r: (
            priority_order[r["priority"]],
            -r["required_cost"],
            r["feed_name"].lower(),
        )
    )

    remaining_budget = budget
    for item in recommendations:
        required = item["required_order_kg"]
        unit_cost = item["unit_cost"]

        if remaining_budget is None:
            allocated = required
        elif unit_cost == 0:
            allocated = required
        else:
            affordable = math.floor(
                (remaining_budget / unit_cost) * 100
            ) / 100
            allocated = min(required, max(0.0, affordable))

        item["recommended_order_kg"] = round(allocated, 2)
        item["estimated_purchase_cost"] = round(
            allocated * unit_cost, 2
        )
        item["budget_limited"] = allocated < required

        if item["budget_limited"]:
            item["budget_note"] = (
                "Full requirement is not affordable under the "
                "remaining budget; review the unmet quantity."
            )
        else:
            item["budget_note"] = "Full calculated requirement allocated."

        if remaining_budget is not None:
            remaining_budget = max(
                0.0,
                remaining_budget - item["estimated_purchase_cost"],
            )

        if item["recommended_order_kg"] > 0 and item["unit_cost"] > 0:
            order_days = (
                item["days_of_stock_remaining"]
                - item["supplier_lead_time_days"]
            )
            if order_days <= 0:
                item["suggested_order_date"] = date.today().isoformat()
            else:
                item["suggested_order_date"] = (
                    date.today()
                    + timedelta(days=max(0, math.floor(order_days)))
                ).isoformat()
        else:
            item["suggested_order_date"] = None

    return {
        "planning_horizon_days": planning_horizon_days,
        "safety_stock_days": safety_stock_days,
        "budget": budget,
        "total_required_cost": round(
            sum(r["required_cost"] for r in recommendations), 2
        ),
        "total_allocated_cost": round(
            sum(r["estimated_purchase_cost"] for r in recommendations), 2
        ),
        "remaining_budget": (
            round(remaining_budget, 2)
            if remaining_budget is not None else None
        ),
        "recommendations": recommendations,
    }
