from typing import Any, Dict, List


def calculate_herd_feed_demand(
    animal_dmi: List[Dict[str, Any]],
    planning_days: int = 7,
) -> Dict[str, Any]:
    """
    Estimate herd dry-matter demand from one selected DMI value per animal.

    Each record must contain:
        animal_id
        predicted_dmi_kg_day

    Only valid, non-negative predictions are included.
    """
    if planning_days < 1:
        raise ValueError("planning_days must be at least 1")

    latest_by_animal = {}

    for record in animal_dmi:
        animal_id = str(record.get("animal_id", "")).strip()

        if not animal_id:
            continue

        try:
            dmi = float(record["predicted_dmi_kg_day"])
        except (KeyError, TypeError, ValueError):
            continue

        if dmi < 0:
            continue

        # Input must contain at most one selected prediction per animal.
        if animal_id in latest_by_animal:
            raise ValueError(
                f"Multiple DMI predictions supplied for animal {animal_id}"
            )

        latest_by_animal[animal_id] = dmi

    daily_demand = sum(latest_by_animal.values())

    return {
        "animals_included": len(latest_by_animal),
        "daily_demand_kg": round(daily_demand, 3),
        "weekly_demand_kg": round(daily_demand * 7, 3),
        "planning_days": planning_days,
        "planned_demand_kg": round(daily_demand * planning_days, 3),
        "animal_dmi": latest_by_animal,
    }
