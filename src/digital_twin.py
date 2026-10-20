"""
Digital Twin domain model, multimodal risk assessment, and explainable health reporting.
Reuses and extends algorithms from the experimental notebook.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from src.database import (
    get_animal,
    get_observations,
    get_latest_observation,
    get_latest_sensor_reading,
    get_latest_risk_assessment,
    add_observation,
    add_sensor_reading,
    save_risk_assessment,
    add_dmi_prediction,
    get_dmi_predictions,
    get_latest_dmi_prediction,
    init_db,
)


def get_digital_twin(animal_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve and construct the complete Digital Twin state for a given animal ID.
    Directly conforms to the dictionary format established in notebookef7b497bba (1).ipynb.
    """
    # Ensure DB is initialized
    init_db(db_path)

    animal = get_animal(animal_id, db_path=db_path)
    if not animal:
        return None

    observations = get_observations(animal_id, limit=50, db_path=db_path)
    latest_obs = get_latest_observation(animal_id, db_path=db_path)
    latest_sensor = get_latest_sensor_reading(animal_id, db_path=db_path)
    latest_risk = get_latest_risk_assessment(animal_id, db_path=db_path)
    latest_dmi = get_latest_dmi_prediction(animal_id, db_path=db_path)
    dmi_history = get_dmi_predictions(animal_id, limit=50, db_path=db_path)

    # Current state
    if latest_obs:
        current_state = {
            "timestamp": latest_obs["timestamp"],
            "predicted_condition": latest_obs["cv_prediction"],
            "cv_confidence": float(latest_obs["cv_confidence"]),
            "model_used": latest_obs.get("model_used", "DenseNet-121"),
        }
    else:
        current_state = {
            "timestamp": animal["created_at"],
            "predicted_condition": "Unknown",
            "cv_confidence": 0.0,
            "model_used": "N/A",
        }

    # History list
    history = [
        {
            "timestamp": obs["timestamp"],
            "observation_type": obs["observation_type"],
            "cv_prediction": obs["cv_prediction"],
            "cv_confidence": float(obs["cv_confidence"]),
            "model_used": obs.get("model_used"),
            "notes": obs.get("notes"),
        }
        for obs in observations
    ]

    # Sensor data
    sensor_data = {}
    environment_data = {}
    if latest_sensor:
        sensor_data = {
            "body_temperature_c": latest_sensor["body_temperature_c"],
            "activity_level": latest_sensor["activity_level"],
            "feed_intake": latest_sensor["feed_intake"],
            "data_source": latest_sensor["data_source"],
            "is_simulated": bool(latest_sensor["is_simulated"]),
            "timestamp": latest_sensor["timestamp"],
        }
        environment_data = {
            "ambient_temperature_c": latest_sensor["ambient_temperature_c"],
            "humidity_percent": latest_sensor["humidity_percent"],
            "data_source": latest_sensor["data_source"],
            "is_simulated": bool(latest_sensor["is_simulated"]),
            "timestamp": latest_sensor["timestamp"],
        }

    # Risk assessment
    if latest_risk:
        risk_assessment = {
            "risk_score": latest_risk["risk_score"],
            "status": latest_risk["risk_status"],
            "reasons": latest_risk["reasons"],
            "timestamp": latest_risk["timestamp"],
        }
    else:
        risk_assessment = {
            "risk_score": 0,
            "status": "Not assessed",
            "reasons": ["No risk assessment recorded yet."],
            "timestamp": animal["created_at"],
        }

    digital_twin = {
        "animal_id": animal["animal_id"],
        "species": animal["species"],
        "name": animal.get("name") or "Unnamed",
        "breed": animal.get("breed") or "Unspecified",
        "birth_date": animal.get("birth_date"),
        "created_at": animal["created_at"],
        "updated_at": animal["updated_at"],
        "current_state": current_state,
        "history": history,
        "sensor_data": sensor_data,
        "environment_data": environment_data,
        "risk_assessment": risk_assessment,
        "latest_dmi": latest_dmi,
        "dmi_history": dmi_history,
    }

    return digital_twin


def multimodal_risk_assessment(digital_twin: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multimodal rule-based health-risk assessment algorithm.
    Directly reuses the logic from notebookef7b497bba (1).ipynb (Cell 29).
    
    Scoring breakdown:
    - Current CV disease: +4 points
    - Historical CV disease: +1 point
    - Elevated body temperature (>= 39.0°C): +1 point
    - Low activity level: +1 point
    - Reduced feed intake: +1 point
    - High ambient temperature (>= 30.0°C): +1 point
    - High humidity (>= 70.0%): +1 point

    Classification:
    - Score >= 6: High Risk
    - Score >= 3: Moderate Risk
    - Score < 3: Low Risk
    """
    score = 0
    reasons: List[str] = []

    # 1. Current CV observation
    current_condition = (
        digital_twin.get("current_state", {})
        .get("predicted_condition", "")
        .lower()
    )

    if current_condition in ["lumpy", "foot-and-mouth"]:
        score += 4
        reasons.append(f"Current CV observation indicates {current_condition}.")

    # 2. Previous disease history
    history = digital_twin.get("history", [])
    if len(history) > 1:
        previous_conditions = [
            obs.get("cv_prediction", "").lower()
            for obs in history[:-1]
        ]
        if any(c in ["lumpy", "foot-and-mouth"] for c in previous_conditions):
            score += 1
            reasons.append("Previous observation indicated disease.")

    # 3. Body temperature
    sensor_data = digital_twin.get("sensor_data", {})
    temperature = sensor_data.get("body_temperature_c")
    if temperature is not None and float(temperature) >= 39.0:
        score += 1
        reasons.append("Body temperature is elevated in this prototype scenario.")

    # 4. Activity
    activity = sensor_data.get("activity_level")
    if activity == "Low":
        score += 1
        reasons.append("Low activity level was recorded.")

    # 5. Feed intake
    feed = sensor_data.get("feed_intake")
    if feed == "Reduced":
        score += 1
        reasons.append("Reduced feed intake was recorded.")

    # 6. Environmental conditions
    env_data = digital_twin.get("environment_data", {})
    ambient_temp = env_data.get("ambient_temperature_c")
    humidity = env_data.get("humidity_percent")

    if ambient_temp is not None and float(ambient_temp) >= 30.0:
        score += 1
        reasons.append("High ambient temperature provides additional environmental context.")

    if humidity is not None and float(humidity) >= 70.0:
        score += 1
        reasons.append("High humidity provides additional environmental context.")

    # Final prototype risk category
    if score >= 6:
        risk_status = "High"
    elif score >= 3:
        risk_status = "Moderate"
    else:
        risk_status = "Low"
        if not reasons:
            reasons.append("No disease condition or physiological abnormalities detected in recorded history.")

    digital_twin["risk_assessment"] = {
        "risk_score": score,
        "status": risk_status,
        "reasons": reasons,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return digital_twin


def generate_health_report(digital_twin: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an explainable health report structure directly conforming to
    notebookef7b497bba (1).ipynb (Cell 30).
    """
    current = digital_twin.get("current_state", {})
    risk = digital_twin.get("risk_assessment", {})
    sensors = digital_twin.get("sensor_data", {})
    environment = digital_twin.get("environment_data", {})

    conf = current.get("cv_confidence", 0.0)
    # If conf is in [0, 1], scale to percentage; if already > 1, keep as is
    conf_scaled = round(conf * 100 if conf <= 1.0 else conf, 2)

    historical_context = []
    for obs in digital_twin.get("history", []):
        obs_conf = obs.get("cv_confidence", 0.0)
        historical_context.append({
            "timestamp": obs.get("timestamp"),
            "condition": obs.get("cv_prediction"),
            "confidence": round(obs_conf * 100 if obs_conf <= 1.0 else obs_conf, 2),
        })

    report = {
        "animal_id": digital_twin.get("animal_id"),
        "species": digital_twin.get("species", "Cattle"),
        "name": digital_twin.get("name", "Unnamed"),
        "current_health_observation": {
            "condition": current.get("predicted_condition"),
            "cv_confidence": conf_scaled,
        },
        "historical_context": historical_context,
        "sensor_observations": sensors,
        "environmental_context": environment,
        "risk_assessment": {
            "status": risk.get("status", "Not assessed"),
            "score": risk.get("risk_score", 0),
            "contributing_factors": risk.get("reasons", []),
        },
    }

    return report


def generate_explainable_summary(digital_twin: Dict[str, Any]) -> Dict[str, str]:
    """
    Construct human-readable, grounded explainable summary sections based on
    the 10 rules from notebook cell 32:
    - Animal Summary
    - Current Observation
    - Historical Trend
    - Contributing Factors
    - Risk Assessment
    - Recommended Follow-up
    """
    report = generate_health_report(digital_twin)
    curr = report["current_health_observation"]
    risk = report["risk_assessment"]
    history = report["historical_context"]
    sensors = report["sensor_observations"]
    env = report["environmental_context"]

    # 1. Animal Summary
    summary = (
        f"Animal ID {digital_twin['animal_id']} ({digital_twin.get('name', 'N/A')}), "
        f"Species: {digital_twin.get('species', 'Cattle')}, Breed: {digital_twin.get('breed', 'Unspecified')}. "
        f"Total recorded health observations: {len(history)}."
    )

    # 2. Current Observation
    condition = curr.get("condition", "Unknown")
    conf = curr.get("cv_confidence", 0.0)
    current_obs_text = (
        f"Computer-vision image screening estimated the animal's condition as '{condition}' "
        f"with {conf:.2f}% confidence."
    )

    # 3. Historical Trend
    if len(history) <= 1:
        trend_text = "Baseline record established. No earlier observations in history."
    else:
        prev_conditions = [h["condition"] for h in history[:-1]]
        last_condition = prev_conditions[-1]
        if last_condition == condition:
            trend_text = f"Consistent condition trend: Animal was previously evaluated as '{last_condition}'."
        else:
            trend_text = (
                f"Condition change detected: Previous observation was '{last_condition}', "
                f"while the latest observation transitioned to '{condition}'."
            )

    # 4. Contributing Factors
    factors_list = risk.get("contributing_factors", [])
    if factors_list:
        factors_text = "\n".join([f"• {f}" for f in factors_list])
    else:
        factors_text = "• No adverse physiological or environmental factors recorded."

    # 5. Risk Assessment
    risk_text = (
        f"Prototype Multimodal Risk Score: {risk.get('score', 0)}/10 ({risk.get('status', 'Unknown')} Risk). "
        f"This status reflects combined computer-vision findings and available physiological/environmental inputs."
    )

    # 6. Recommended Follow-up
    if risk.get("status") == "High" or condition in ["foot-and-mouth", "lumpy"]:
        follow_up_text = (
            "URGENT: Isolate the animal immediately from the herd to prevent disease transmission. "
            "Arrange prompt clinical examination by a licensed veterinarian and obtain confirmatory laboratory "
            "diagnostics (e.g. RT-PCR, serology). Monitor hydration, feed intake, and rectal temperature closely."
        )
    elif risk.get("status") == "Moderate":
        follow_up_text = (
            "MONITOR: Maintain active surveillance. Re-evaluate body temperature and feed intake every 12 hours. "
            "If clinical signs progress or body temperature exceeds 39.5°C, schedule veterinary consultation."
        )
    else:
        follow_up_text = (
            "ROUTINE: Continue standard herd management, biosecurity protocols, and scheduled screening."
        )

    return {
        "animal_summary": summary,
        "current_observation": current_obs_text,
        "historical_trend": trend_text,
        "contributing_factors": factors_text,
        "risk_assessment": risk_text,
        "recommended_follow_up": follow_up_text,
    }


def record_cv_observation(
    animal_id: str,
    cv_prediction: str,
    cv_confidence: float,
    model_used: str = "DenseNet-121",
    observation_type: str = "image",
    notes: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Add a new CV observation to an animal's digital twin, recalculate multimodal risk,
    and persist the updated state.
    """
    add_observation(
        animal_id=animal_id,
        cv_prediction=cv_prediction,
        cv_confidence=cv_confidence,
        model_used=model_used,
        observation_type=observation_type,
        notes=notes,
        timestamp=timestamp,
        db_path=db_path,
    )

    # Fetch updated twin
    dt = get_digital_twin(animal_id, db_path=db_path)
    if dt is None:
        raise ValueError(f"Animal {animal_id} not found.")

    # Recalculate risk
    dt = multimodal_risk_assessment(dt)
    risk_info = dt["risk_assessment"]
    save_risk_assessment(
        animal_id=animal_id,
        risk_score=risk_info["risk_score"],
        risk_status=risk_info["status"],
        reasons=risk_info["reasons"],
        timestamp=risk_info["timestamp"],
        db_path=db_path,
    )

    return dt


def record_sensor_reading(
    animal_id: str,
    body_temperature_c: Optional[float],
    activity_level: Optional[str],
    feed_intake: Optional[str],
    ambient_temperature_c: Optional[float],
    humidity_percent: Optional[float],
    data_source: str = "Manual Entry",
    is_simulated: bool = False,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Add a sensor / environmental reading to an animal's digital twin,
    recalculate multimodal risk, and persist the updated state.
    """
    add_sensor_reading(
        animal_id=animal_id,
        body_temperature_c=body_temperature_c,
        activity_level=activity_level,
        feed_intake=feed_intake,
        ambient_temperature_c=ambient_temperature_c,
        humidity_percent=humidity_percent,
        data_source=data_source,
        is_simulated=is_simulated,
        timestamp=timestamp,
        db_path=db_path,
    )

    # Fetch updated twin
    dt = get_digital_twin(animal_id, db_path=db_path)
    if dt is None:
        raise ValueError(f"Animal {animal_id} not found.")

    # Recalculate risk
    dt = multimodal_risk_assessment(dt)
    risk_info = dt["risk_assessment"]
    save_risk_assessment(
        animal_id=animal_id,
        risk_score=risk_info["risk_score"],
        risk_status=risk_info["status"],
        reasons=risk_info["reasons"],
        timestamp=risk_info["timestamp"],
        db_path=db_path,
    )

    return dt


def record_dmi_prediction(
    animal_id: str,
    predicted_dmi_kg_day: float,
    model_name: str,
    input_features: Dict[str, Any],
    notes: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Record an experimental daily DMI prediction for an animal's digital twin
    and persist the updated state.
    """
    add_dmi_prediction(
        animal_id=animal_id,
        predicted_dmi_kg_day=predicted_dmi_kg_day,
        model_name=model_name,
        input_features=input_features,
        notes=notes,
        timestamp=timestamp,
        db_path=db_path,
    )

    dt = get_digital_twin(animal_id, db_path=db_path)
    if dt is None:
        raise ValueError(f"Animal {animal_id} not found.")

    return dt

