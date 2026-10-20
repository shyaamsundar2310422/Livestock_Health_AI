import pytest

from src.procurement_optimizer import (
    calculate_ration_demand,
    optimize_procurement,
    resolve_saved_herd_dmi,
)


def sample_inventory():
    return [
        {
            "feed_name": "Maize",
            "available_stock_kg": 30,
            "daily_usage_kg": 10,
            "unit_cost": 2,
            "supplier_name": "Supplier A",
            "supplier_lead_time_days": 2,
            "reorder_threshold_kg": 20,
        },
        {
            "feed_name": "Wheat",
            "available_stock_kg": 80,
            "daily_usage_kg": 4,
            "unit_cost": 2,
            "supplier_name": "Supplier B",
            "supplier_lead_time_days": 2,
            "reorder_threshold_kg": 30,
        },
    ]


def test_urgent_feed_has_purchase_requirement():
    result = optimize_procurement(
        sample_inventory(),
        planning_horizon_days=7,
        safety_stock_days=3,
        budget=None,
    )

    maize = next(
        item for item in result["recommendations"]
        if item["feed_name"] == "Maize"
    )

    assert maize["required_order_kg"] == pytest.approx(70)
    assert maize["recommended_order_kg"] == pytest.approx(70)
    assert maize["required_cost"] == pytest.approx(140)


def test_budget_limit_is_reported():
    result = optimize_procurement(
        sample_inventory(),
        planning_horizon_days=7,
        safety_stock_days=3,
        budget=100,
    )

    assert result["total_allocated_cost"] <= 100
    assert result["remaining_budget"] == pytest.approx(0)

    maize = next(
        item for item in result["recommendations"]
        if item["feed_name"] == "Maize"
    )

    assert maize["budget_limited"] is True
    assert maize["recommended_order_kg"] < maize["required_order_kg"]


def test_sufficient_stock_does_not_require_order():
    result = optimize_procurement(
        sample_inventory(),
        planning_horizon_days=7,
        safety_stock_days=3,
        budget=None,
    )

    wheat = next(
        item for item in result["recommendations"]
        if item["feed_name"] == "Wheat"
    )

    assert wheat["required_order_kg"] == pytest.approx(0)
    assert wheat["recommended_order_kg"] == pytest.approx(0)


def test_invalid_planning_horizon_is_rejected():
    with pytest.raises(ValueError):
        optimize_procurement(
            sample_inventory(),
            planning_horizon_days=0,
        )


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        optimize_procurement(
            sample_inventory(),
            budget=-1,
        )


def test_saved_dmi_sums_latest_valid_predictions_and_reports_missing_or_invalid():
    animals = [
        {"animal_id": "C001"},
        {"animal_id": "C002"},
        {"animal_id": "C003"},
        {"animal_id": "C004"},
    ]
    latest = {
        "C001": {"predicted_dmi_kg_day": 21.5},
        "C002": {"predicted_dmi_kg_day": 19.0},
        "C003": {"predicted_dmi_kg_day": float("nan")},
        "C004": None,
    }

    result = resolve_saved_herd_dmi(animals, latest.get)

    assert result["animal_dmi_kg_day"] == {"C001": 21.5, "C002": 19.0}
    assert result["total_dmi_kg_day"] == pytest.approx(40.5)
    assert result["issues"] == [
        "C003: latest DMI prediction must be finite and greater than zero",
        "C004: no saved DMI prediction",
    ]


def test_ration_shares_drive_daily_horizon_demand_and_procurement_cost():
    items = sample_inventory()
    shares = {"Maize": 70, "Wheat": 30}

    demand = calculate_ration_demand(items, 100, shares, planning_horizon_days=10)
    maize = next(row for row in demand if row["feed_name"] == "Maize")
    wheat = next(row for row in demand if row["feed_name"] == "Wheat")

    assert maize["daily_demand_kg"] == pytest.approx(70)
    assert wheat["daily_demand_kg"] == pytest.approx(30)
    assert maize["horizon_demand_kg"] == pytest.approx(700)
    assert wheat["horizon_demand_kg"] == pytest.approx(300)

    procurement_inventory = [
        {**row, "daily_usage_kg": row["daily_demand_kg"]}
        for row in demand
    ]
    result = optimize_procurement(
        procurement_inventory,
        planning_horizon_days=7,
        safety_stock_days=3,
        budget=None,
    )
    maize_order = next(row for row in result["recommendations"] if row["feed_name"] == "Maize")
    wheat_order = next(row for row in result["recommendations"] if row["feed_name"] == "Wheat")
    assert maize_order["required_order_kg"] == pytest.approx(670)
    assert wheat_order["required_order_kg"] == pytest.approx(220)
    assert result["total_required_cost"] == pytest.approx(1780)


def test_ration_demand_rejects_invalid_totals():
    with pytest.raises(ValueError, match="total 100%"):
        calculate_ration_demand(
            sample_inventory(),
            100,
            {"Maize": 40, "Wheat": 40},
            planning_horizon_days=7,
        )
