from src.feed_demand import calculate_herd_feed_demand


def test_herd_demand():
    result = calculate_herd_feed_demand([
        {"animal_id": "C001", "predicted_dmi_kg_day": 18.0},
        {"animal_id": "C002", "predicted_dmi_kg_day": 21.5},
        {"animal_id": "C003", "predicted_dmi_kg_day": 20.0},
    ])

    assert result["animals_included"] == 3
    assert result["daily_demand_kg"] == 59.5
    assert result["weekly_demand_kg"] == 416.5
    assert result["planned_demand_kg"] == 416.5


if __name__ == "__main__":
    test_herd_demand()
    print("Feed demand calculation passed.")
