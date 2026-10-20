"""
Streamlit Web Application for Cattle Disease Classification, Digital Twins, and Explainability.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image

import streamlit as st
import pandas as pd
from src.procurement_optimizer import (
    calculate_ration_demand,
    optimize_procurement,
    resolve_saved_herd_dmi,
)

# Add project root to system path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CLASS_KEYS,
    CLASS_DISPLAY_NAMES,
    CLASS_DESCRIPTIONS,
    AVAILABLE_MODELS,
    VETERINARY_DISCLAIMER,
    SAMPLE_IMAGES_DIR,
    SPECIES_LIST,
    ACTIVITY_LEVELS,
    FEED_INTAKE_LEVELS,
    NORMAL_BODY_TEMP_MIN,
    NORMAL_BODY_TEMP_MAX,
    DMI_MODEL_NAME,
    DMI_FEATURE_NAMES,
    DMI_FEATURE_METADATA,
    DMI_DISCLAIMER,
)
from src.model import load_model, predict, validate_and_load_image, get_device
from src.gradcam import generate_gradcam_visualizations
from src.database import (
    init_db,
    list_animals,
    create_animal,
    delete_animal,
    get_observations,
    get_sensor_readings,
    get_dmi_predictions,
    get_feed_inventory,
    upsert_feed_inventory_item,
    get_latest_dmi_prediction,
)
from src.digital_twin import (
    get_digital_twin,
    record_cv_observation,
    record_sensor_reading,
    record_dmi_prediction,
    generate_explainable_summary,
)
from src.dmi_model import (
    check_dmi_model_availability,
    predict_dmi,
)
from src.feed_inventory import calculate_inventory_status

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Livestock Health AI - Disease Classifier & Digital Twin",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database
init_db()

# Custom CSS for polished styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.85rem 1.1rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .result-card-healthy {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card-fmd {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card-lumpy {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .tag-simulated {
        background-color: #E0E7FF;
        color: #3730A3;
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .tag-manual {
        background-color: #F3F4F6;
        color: #374151;
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Header & Veterinary Disclaimer
# ---------------------------------------------------------
st.markdown('<div class="main-header">🐄 Livestock Health AI: Disease Classifier & Digital Twin</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Integrated computer vision disease screening, individual animal Digital Twins, '
    'multimodal risk assessment, and Grad-CAM explainability.</div>',
    unsafe_allow_html=True,
)

st.markdown(f'<div class="disclaimer-box">{VETERINARY_DISCLAIMER}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("🏷️ Active Animal Profile")

all_animals = list_animals()
animal_options = {f"{a['animal_id']} - {a.get('name') or 'Unnamed'} ({a.get('species')})": a["animal_id"] for a in all_animals}
animal_keys = ["None (Unlinked Screening)"] + list(animal_options.keys())

selected_animal_display = st.sidebar.selectbox(
    "Select Animal Digital Twin:",
    options=animal_keys,
    index=1 if len(animal_keys) > 1 else 0,
    help="Link computer-vision predictions and sensor telemetry to this individual animal's Digital Twin.",
)

active_animal_id = animal_options.get(selected_animal_display)

if active_animal_id:
    twin_data = get_digital_twin(active_animal_id)
    if twin_data:
        curr_cond = twin_data["current_state"].get("predicted_condition", "Unknown")
        risk_status = twin_data["risk_assessment"].get("status", "Not assessed")
        score = twin_data["risk_assessment"].get("risk_score", 0)

        st.sidebar.markdown(f"**Species / Breed:** `{twin_data.get('species')} / {twin_data.get('breed')}`")
        st.sidebar.markdown(f"**Current Condition:** `{curr_cond.capitalize()}`")
        
        if risk_status == "High":
            st.sidebar.markdown(f"**Risk Level:** <span class='risk-high'>HIGH (Score: {score})</span>", unsafe_allow_html=True)
        elif risk_status == "Moderate":
            st.sidebar.markdown(f"**Risk Level:** <span class='risk-moderate'>MODERATE (Score: {score})</span>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"**Risk Level:** <span class='risk-low'>LOW (Score: {score})</span>", unsafe_allow_html=True)

st.sidebar.divider()

# Model Selector
st.sidebar.subheader("⚙️ Vision Model Settings")
model_options = list(AVAILABLE_MODELS.keys())
selected_model_label = st.sidebar.selectbox(
    "Select Classification Model:",
    options=model_options,
    index=0,
    help="DenseNet-121 is recommended as it achieved the highest test accuracy (96.30%) across all benchmarks.",
)

model_info = AVAILABLE_MODELS[selected_model_label]
model_id = model_info["id"]

st.sidebar.markdown(f"**Test Accuracy:** `{model_info['test_accuracy']}`")
st.sidebar.markdown(f"**Macro F1-Score:** `{model_info['macro_f1']}`")

# Explainability Settings
st.sidebar.subheader("🔍 Explainable AI (XAI)")
enable_gradcam = st.sidebar.checkbox("Enable Grad-CAM Visual Heatmap", value=True)
blend_alpha = st.sidebar.slider(
    "Heatmap Blend Opacity",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    disabled=not enable_gradcam,
)

st.sidebar.divider()

# Sample image quick test
st.sidebar.subheader("🧪 Demo Cattle Images")
sample_files = {
    "None (Upload your own)": None,
    "Healthy Cow (Pasture)": "healthy_sample2.jpg",
    "Healthy Cow (Farm)": "healthy_sample1.jpg",
    "Foot-and-Mouth Disease (Lesions 1)": "foot_and_mouth_sample1.jpg",
    "Foot-and-Mouth Disease (Lesions 2)": "foot_and_mouth_sample2.jpg",
    "Lumpy Skin Disease (Nodules 1)": "lumpy_skin_sample1.jpg",
    "Lumpy Skin Disease (Nodules 2)": "lumpy_skin_sample2.jpg",
}

selected_sample_label = st.sidebar.selectbox(
    "Load Preloaded Sample Image:",
    options=list(sample_files.keys()),
    index=0,
)

# System Info in Sidebar
device = get_device()
st.sidebar.divider()
st.sidebar.caption(f"**Hardware Device:** `{device.type.upper()}`")
st.sidebar.caption("**Input Resolution:** `224 × 224 px`")
st.sidebar.caption("**Database:** `SQLite (local)`")

# ---------------------------------------------------------
# Main Application Tabs
# ---------------------------------------------------------
tab_diag, tab_twin, tab_dmi, tab_feed, tab_history, tab_benchmarks, tab_guide = st.tabs([
    "🩺 Disease Diagnosis",
    "🧬 Animal Digital Twins",
    "🌾 Dry Matter Intake (DMI)",
    "📦 Feed Supply Chain",
    "📈 Health History & Dashboard",
    "📊 Model Comparison & Benchmarks",
    "📚 Clinical Disease Guide",
])

# =========================================================================
# TAB 1: DISEASE DIAGNOSIS (Preserved and Enhanced with Digital Twin integration)
# =========================================================================
with tab_diag:
    col_input, col_output = st.columns([1, 1], gap="large")

    image_to_process = None
    image_source_desc = ""

    with col_input:
        st.subheader("1. Input Cattle Image")
        
        uploaded_file = st.file_uploader(
            "Upload a clear image of cattle (face, muzzle, limbs, or body skin):",
            type=["jpg", "jpeg", "png", "webp"],
            help="High-quality, close-up photos of lesions or skin nodules produce the most accurate predictions.",
        )

        if uploaded_file is not None:
            try:
                image_to_process = validate_and_load_image(uploaded_file)
                image_source_desc = f"Uploaded File: {uploaded_file.name}"
            except ValueError as e:
                st.error(f"❌ Invalid Image File: {str(e)}")
        elif sample_files[selected_sample_label] is not None:
            sample_rel_path = SAMPLE_IMAGES_DIR / sample_files[selected_sample_label]
            if sample_rel_path.exists():
                try:
                    image_to_process = validate_and_load_image(sample_rel_path)
                    image_source_desc = f"Sample: {selected_sample_label}"
                except Exception as e:
                    st.error(f"Failed to load sample image: {e}")
            else:
                st.warning(f"Sample image not found at {sample_rel_path}")

        if image_to_process is not None:
            st.image(
                image_to_process,
                caption=f"Input Image ({image_source_desc}) - Size: {image_to_process.size[0]}×{image_to_process.size[1]} px",
                width="stretch",
            )
            run_btn = st.button("🚀 Analyze Cattle Health", type="primary", use_container_width=True)
        else:
            st.info("👆 Please upload an image or select a preloaded demo sample from the sidebar to begin analysis.")
            run_btn = False

    with col_output:
        st.subheader("2. AI Diagnosis Results")

        if image_to_process is not None and (run_btn or st.session_state.get("auto_run", True)):
            st.session_state["auto_run"] = True

            with st.spinner(f"Analyzing image using {selected_model_label}..."):
                try:
                    pred_res = predict(image_to_process, model_name=model_id, device=device)
                    pred_key = pred_res["predicted_key"]
                    pred_class = pred_res["predicted_class"]
                    conf = pred_res["confidence"]
                    conf_pct = pred_res["confidence_percent"]
                    probs = pred_res["probabilities"]

                    # Display Status Card
                    if pred_key == "healthy":
                        card_class = "result-card-healthy"
                        status_badge = "✅ HEALTHY CATTLE"
                    elif pred_key == "foot-and-mouth":
                        card_class = "result-card-fmd"
                        status_badge = "🚨 FOOT-AND-MOUTH DISEASE DETECTED"
                    else:
                        card_class = "result-card-lumpy"
                        status_badge = "⚠️ LUMPY SKIN DISEASE DETECTED"

                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <h2 style="margin:0; font-size: 1.5rem;">{status_badge}</h2>
                            <p style="margin: 0.3rem 0 0 0; font-size: 1.1rem; color: #374151;">
                                Predicted Class: <b>{pred_class}</b>
                            </p>
                            <h3 style="margin: 0.3rem 0 0 0; color: #1F2937;">Confidence: {conf_pct}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Metrics Columns
                    m1, m2, m3 = st.columns(3)
                    for idx, (cls_name, prob_val) in enumerate(probs.items()):
                        col = [m1, m2, m3][idx]
                        col.metric(
                            label=cls_name,
                            value=f"{prob_val * 100:.1f}%",
                        )

                    # Probability Breakdown Bar Chart
                    st.write("**Probability Distribution:**")
                    prob_df = pd.DataFrame(
                        {
                            "Condition": list(probs.keys()),
                            "Probability": list(probs.values()),
                        }
                    ).set_index("Condition")
                    st.bar_chart(prob_df, y="Probability", color="#2563EB")

                    # Integration with Digital Twin
                    st.divider()
                    st.write("### 💾 Record into Digital Twin")
                    
                    target_animal_opts = ["-- Choose Animal Profile --"] + [
                        f"{a['animal_id']} - {a.get('name') or 'Unnamed'}" for a in all_animals
                    ]
                    default_idx = 0
                    if active_animal_id:
                        for i, opt in enumerate(target_animal_opts[1:], start=1):
                            if opt.startswith(active_animal_id):
                                default_idx = i
                                break

                    selected_target_animal = st.selectbox(
                        "Target Animal Profile:",
                        options=target_animal_opts,
                        index=default_idx,
                        key="dt_save_select",
                    )

                    obs_notes = st.text_input(
                        "Observation Notes (Optional):",
                        placeholder="e.g. Visual screening during morning inspection",
                        key="dt_obs_notes",
                    )

                    if st.button("💾 Save Observation to Digital Twin", type="secondary", use_container_width=True):
                        if selected_target_animal == "-- Choose Animal Profile --":
                            st.warning("Please select an animal profile to save this observation.")
                        else:
                            save_id = selected_target_animal.split(" - ")[0].strip()
                            updated_twin = record_cv_observation(
                                animal_id=save_id,
                                cv_prediction=pred_key,
                                cv_confidence=conf,
                                model_used=model_info["id"],
                                notes=obs_notes,
                            )
                            st.success(
                                f"✅ Successfully logged prediction into Digital Twin **{save_id}**! "
                                f"Updated Multimodal Risk Status: **{updated_twin['risk_assessment']['status']}** "
                                f"(Score: {updated_twin['risk_assessment']['risk_score']})"
                            )
                            st.rerun()

                    # Clinical Description
                    with st.expander("📖 Clinical Description & Disease Significance", expanded=(pred_key != "healthy")):
                        st.write(CLASS_DESCRIPTIONS[pred_key])
                        if pred_key != "healthy":
                            st.warning(
                                "⚠️ Veterinary Action Protocol: Isolate the animal immediately from the herd "
                                "and notify veterinary authorities for confirmatory laboratory testing (PCR/ELISA)."
                            )

                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")

        elif image_to_process is None:
            st.write("Results and explainability visualizations will appear here after an image is selected.")

    # Grad-CAM Section
    if image_to_process is not None and enable_gradcam and st.session_state.get("auto_run", False):
        st.divider()
        st.subheader("3. Explainable AI: Grad-CAM Saliency Analysis")
        st.write(
            "Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes the specific visual regions "
            "and texture features that most influenced the convolutional network's decision."
        )

        with st.spinner("Generating Grad-CAM visualization..."):
            try:
                grad_data = generate_gradcam_visualizations(
                    image=image_to_process,
                    model_name=model_id,
                    alpha=blend_alpha,
                    device=device,
                )

                gc_col1, gc_col2, gc_col3 = st.columns(3)

                with gc_col1:
                    st.image(image_to_process, caption="1. Original Input Image", width="stretch")

                with gc_col2:
                    st.image(grad_data["heatmap"], caption="2. Activation Heatmap (Jet Colormap)", width="stretch")

                with gc_col3:
                    st.image(
                        grad_data["overlay"],
                        caption=f"3. Grad-CAM Overlay ({int(blend_alpha*100)}% blend) for {grad_data['target_class_name']}",
                        width="stretch",
                    )

                st.caption(
                    "🔴 **Red & Yellow Regions:** High network activation / key lesion markers driving the prediction. "
                    "🔵 **Blue Regions:** Low network influence / background."
                )

            except Exception as e:
                st.error(f"Grad-CAM Generation Error: {str(e)}")


# =========================================================================
# TAB 2: ANIMAL DIGITAL TWINS (Profiles, Telemetry & Multimodal Risk)
# =========================================================================
with tab_twin:
    st.header("🧬 Individual Animal Digital Twins")
    st.write(
        "Each Digital Twin maintains an individual animal's live profile, computer-vision disease history, "
        "physiological telemetry, environmental context, and an explainable multimodal risk assessment."
    )

    twin_subtab1, twin_subtab2, twin_subtab3 = st.tabs([
        "📋 Live Twin Status & Risk",
        "📡 Log Sensor & Environment Telemetry",
        "➕ Create / Manage Animal Profiles",
    ])

    with twin_subtab1:
        if not active_animal_id:
            st.info("👆 Please select an animal profile from the sidebar to inspect its Digital Twin.")
        else:
            curr_twin = get_digital_twin(active_animal_id)
            if curr_twin:
                st.subheader(f"Digital Twin: {curr_twin['animal_id']} ({curr_twin.get('name')})")

                # Top Metrics Row
                t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
                
                with t_col1:
                    st.metric("Species / Breed", f"{curr_twin.get('species')}", curr_twin.get('breed'))
                
                with t_col2:
                    curr_cond = curr_twin["current_state"].get("predicted_condition", "Unknown")
                    conf_val = curr_twin["current_state"].get("cv_confidence", 0.0)
                    conf_disp = f"{conf_val * 100:.1f}%" if conf_val <= 1.0 else f"{conf_val:.1f}%"
                    st.metric("Latest CV Condition", curr_cond.capitalize(), f"Confidence: {conf_disp}")

                with t_col3:
                    sensor_info = curr_twin.get("sensor_data", {})
                    temp_val = sensor_info.get("body_temperature_c")
                    temp_disp = f"{temp_val:.1f} °C" if temp_val is not None else "No data"
                    temp_delta = "Elevated (Fever)" if temp_val and temp_val >= NORMAL_BODY_TEMP_MAX else "Normal"
                    st.metric("Body Temperature", temp_disp, temp_delta)

                with t_col4:
                    risk_info = curr_twin.get("risk_assessment", {})
                    risk_status = risk_info.get("status", "Not assessed")
                    risk_score = risk_info.get("risk_score", 0)
                    st.metric("Multimodal Risk", f"{risk_status} ({risk_score}/10 pts)")
                with t_col5:
                    latest_dmi = curr_twin.get("latest_dmi")
                    if latest_dmi and latest_dmi.get("predicted_dmi_kg_day") is not None:
                        st.metric(
                        "Latest DMI",
                        f"{float(latest_dmi['predicted_dmi_kg_day']):.2f} kg/day",
                    )
                    else:
                        st.metric("Latest DMI", "No data")
                st.divider()

                # Detailed Twin Breakdown
                left_col, right_col = st.columns([1, 1], gap="medium")

                with left_col:
                    st.write("#### 📡 Vital Signs & Telemetry")
                    sensors = curr_twin.get("sensor_data", {})
                    env = curr_twin.get("environment_data", {})

                    if sensors or env:
                        is_sim = sensors.get("is_simulated", False)
                        source_label = sensors.get("data_source", "Manual Entry")
                        tag_html = f"<span class='tag-simulated'>[SIMULATED SENSOR: {source_label}]</span>" if is_sim else f"<span class='tag-manual'>[MANUAL ENTRY: {source_label}]</span>"
                        st.markdown(f"**Data Provenance:** {tag_html}", unsafe_allow_html=True)
                        st.caption(f"Last recorded at: `{sensors.get('timestamp', 'N/A')}`")

                        v_df = pd.DataFrame([
                            {"Parameter": "Body Temperature", "Value": f"{sensors.get('body_temperature_c', 'N/A')} °C", "Normal Reference": "38.0 - 39.0 °C"},
                            {"Parameter": "Activity Level", "Value": str(sensors.get("activity_level", "N/A")), "Normal Reference": "Normal"},
                            {"Parameter": "Feed Intake", "Value": str(sensors.get("feed_intake", "N/A")), "Normal Reference": "Normal"},
                            {"Parameter": "Ambient Temperature", "Value": f"{env.get('ambient_temperature_c', 'N/A')} °C", "Normal Reference": "< 30.0 °C"},
                            {"Parameter": "Relative Humidity", "Value": f"{env.get('humidity_percent', 'N/A')} %", "Normal Reference": "< 70.0 %"},
                        ])
                        st.table(v_df)
                    else:
                        st.info("No sensor or environmental readings recorded yet for this animal.")

                with right_col:
                    st.write("#### 🛡️ Transparent Multimodal Risk Assessment")
                    st.caption("Reused from rule-based framework in research notebook (Cell 29).")

                    risk_status = risk_info.get("status", "Low")
                    risk_score = risk_info.get("risk_score", 0)

                    if risk_status == "High":
                        st.markdown(f"### Status: <span class='risk-high'>HIGH RISK (Score: {risk_score})</span>", unsafe_allow_html=True)
                    elif risk_status == "Moderate":
                        st.markdown(f"### Status: <span class='risk-moderate'>MODERATE RISK (Score: {risk_score})</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### Status: <span class='risk-low'>LOW RISK (Score: {risk_score})</span>", unsafe_allow_html=True)

                    st.write("**Contributing Rule Factors:**")
                    factors = risk_info.get("reasons", [])
                    if factors:
                        for f in factors:
                            st.markdown(f"- 📌 {f}")
                    else:
                        st.write("• No adverse risk factors recorded.")

                    with st.expander("ℹ️ How the Risk Score is Calculated:"):
                        st.markdown(
                            """
                            | Criteria | Points | Condition |
                            | :--- | :---: | :--- |
                            | **Current Vision Disease** | **+4** | Latest observation is FMD or LSD |
                            | **Past Disease History** | **+1** | Earlier observation indicated disease |
                            | **Elevated Temperature** | **+1** | Body temp >= 39.0 °C |
                            | **Low Activity** | **+1** | Activity level marked 'Low' |
                            | **Reduced Feed** | **+1** | Feed intake marked 'Reduced' |
                            | **High Ambient Temp** | **+1** | Ambient temp >= 30.0 °C (heat stress) |
                            | **High Humidity** | **+1** | Relative humidity >= 70.0 % |
                            
                            **Categorization:** Score 0–2: **Low**, 3–5: **Moderate**, 6+: **High**
                            """
                        )

                # Explainable Report Card
                st.divider()
                st.write("### 📝 Explainable Health Report & Follow-up Guidance")
                st.caption("Strictly evidence-grounded summary adhering to veterinary explainability protocols.")
                
                summary_sections = generate_explainable_summary(curr_twin)

                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.markdown("**1. Animal Summary:**")
                    st.write(summary_sections["animal_summary"])
                    st.markdown("**2. Current Observation:**")
                    st.write(summary_sections["current_observation"])
                    st.markdown("**3. Historical Trend:**")
                    st.write(summary_sections["historical_trend"])

                with s_col2:
                    st.markdown("**4. Contributing Factors:**")
                    st.markdown(summary_sections["contributing_factors"])
                    st.markdown("**5. Risk Assessment:**")
                    st.write(summary_sections["risk_assessment"])
                    st.markdown("**6. Recommended Follow-up:**")
                    st.info(summary_sections["recommended_follow_up"])

    with twin_subtab2:
        st.subheader("📡 Record Telemetry / Vital Observations")
        st.write("Add updated physiological sensor measurements and barn environmental conditions.")

        if not active_animal_id:
            st.warning("Please select an active animal profile in the sidebar first.")
        else:
            with st.form("sensor_input_form"):
                st.write(f"Logging data for **{active_animal_id}**:")
                
                f_col1, f_col2 = st.columns(2)

                with f_col1:
                    data_mode = st.radio(
                        "Telemetry Source:",
                        ["Manual Farm Inspection", "Simulated IoT Sensor (Collar/Bolus)"],
                        index=0,
                        help="Clearly designates whether telemetry originates from manual observation or simulated sensor hardware.",
                    )
                    
                    body_temp = st.number_input(
                        "Body Temperature (°C):",
                        min_value=35.0,
                        max_value=43.0,
                        value=38.6,
                        step=0.1,
                        help="Normal bovine body temperature ranges between 38.0°C and 39.0°C.",
                    )

                    activity = st.selectbox("Activity Level:", ACTIVITY_LEVELS, index=0)
                    feed = st.selectbox("Feed Intake:", FEED_INTAKE_LEVELS, index=0)

                with f_col2:
                    ambient_temp = st.number_input(
                        "Barn / Ambient Temperature (°C):",
                        min_value=-10.0,
                        max_value=50.0,
                        value=25.0,
                        step=0.5,
                        help="Temperatures >= 30.0°C indicate heat stress context.",
                    )

                    humidity = st.slider(
                        "Relative Humidity (%):",
                        min_value=10.0,
                        max_value=100.0,
                        value=55.0,
                        step=1.0,
                        help="Humidity >= 70.0% provides environmental disease/heat stress context.",
                    )

                submit_sensor = st.form_submit_button("📥 Commit Telemetry to Digital Twin", type="primary")

                if submit_sensor:
                    is_simulated = "Simulated" in data_mode
                    record_sensor_reading(
                        animal_id=active_animal_id,
                        body_temperature_c=body_temp,
                        activity_level=activity,
                        feed_intake=feed,
                        ambient_temperature_c=ambient_temp,
                        humidity_percent=humidity,
                        data_source=data_mode,
                        is_simulated=is_simulated,
                    )
                    st.success(f"✅ Telemetry committed for {active_animal_id}! Risk assessment recalculated.")
                    st.rerun()

    with twin_subtab3:
        st.subheader("➕ Create New Animal Digital Twin")
        
        with st.form("create_animal_form"):
            c_col1, c_col2 = st.columns(2)
            
            with c_col1:
                new_id = st.text_input("Animal ID (Unique Identifier):", placeholder="e.g. C004").strip()
                new_name = st.text_input("Animal Name / Tag:", placeholder="e.g. Buttercup").strip()
                new_species = st.selectbox("Species:", SPECIES_LIST, index=0)

            with c_col2:
                new_breed = st.text_input("Breed:", placeholder="e.g. Holstein Friesian").strip()
                new_birth = st.date_input("Date of Birth (Approximate):", value=datetime(2023, 1, 1))

            create_btn = st.form_submit_button("✨ Register New Animal Profile", type="primary")

            if create_btn:
                if not new_id:
                    st.error("Animal ID is required.")
                else:
                    try:
                        create_animal(
                            animal_id=new_id,
                            species=new_species,
                            name=new_name if new_name else None,
                            breed=new_breed if new_breed else None,
                            birth_date=str(new_birth),
                        )
                        st.success(f"✅ Animal profile **{new_id}** created successfully!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        st.divider()
        st.write("#### Existing Registered Animal Profiles")
        current_animals = list_animals()
        if current_animals:
            a_df = pd.DataFrame(current_animals)[["animal_id", "species", "name", "breed", "birth_date", "created_at"]]
            st.dataframe(a_df, width="stretch")


# =========================================================================
# TAB 3: DRY MATTER INTAKE (DMI) PREDICTION (INRAE Dataset 2)
# =========================================================================
with tab_dmi:
    st.header("🌾 Daily Dry Matter Intake (DMI) Prediction")
    st.markdown(
        "*Experimental intake prediction based on the Pooled Expanded Features Extra Trees Regressor "
        "(INRAE Dataset 2 - DataBase_Cows.csv).*"
    )

    st.markdown(f'<div class="disclaimer-box">{DMI_DISCLAIMER}</div>', unsafe_allow_html=True)

    # Model status check
    dmi_status = check_dmi_model_availability()
    if dmi_status["is_available"]:
        st.success(f"✅ **Trained Model Active:** {dmi_status['model_name']} (`{dmi_status['model_path']}`)")
    else:
        st.warning(
            f"📦 **Model Artifact Required:** `{dmi_status['model_path']}`\n\n"
            "The prediction engine is scaffolded and ready. To activate live inference, place the trained "
            "Pooled Expanded Extra Trees model package (`.joblib` or `.pkl`) into the `models/` directory. "
            "Input validation and feature engineering are fully operational below."
        )

    st.divider()

    # Animal Association & Autofill
    dmi_col_header1, dmi_col_header2 = st.columns([2, 1])
    with dmi_col_header1:
        st.subheader("1. Cow & Production Feature Inputs")
        if active_animal_id:
            st.caption(f"Currently active animal profile: **{active_animal_id}**")
        else:
            st.caption("No animal profile selected in sidebar. Predictions will run in standalone screening mode.")

    with dmi_col_header2:
        autofill_btn = st.button("📋 Autofill Realistic Reference Values", help="Fill form with typical bovine parameters from INRAE dataset")

    # Form with the exact 9 features
    with st.form("dmi_prediction_form"):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.markdown("##### 🐮 Animal Demographics")
            age_val = st.number_input(
                "Age (months):",
                min_value=12.0,
                max_value=180.0,
                value=48.0 if not autofill_btn else 54.0,
                step=1.0,
                help=DMI_FEATURE_METADATA["age"]["help"],
            )
            parity_val = st.number_input(
                "Parity (Lactation Number):",
                min_value=1.0,
                max_value=10.0,
                value=2.0 if not autofill_btn else 3.0,
                step=1.0,
                help=DMI_FEATURE_METADATA["parity"]["help"],
            )
            dim_val = st.number_input(
                "Days in Milk (DIM):",
                min_value=1.0,
                max_value=500.0,
                value=120.0 if not autofill_btn else 95.0,
                step=1.0,
                help=DMI_FEATURE_METADATA["DIM"]["help"],
            )

        with col_f2:
            st.markdown("##### ⚖️ Body Metrics")
            bw_val = st.number_input(
                "Body Weight (BW, kg):",
                min_value=300.0,
                max_value=1100.0,
                value=620.0 if not autofill_btn else 645.0,
                step=5.0,
                help=DMI_FEATURE_METADATA["BW"]["help"],
            )
            bcs_val = st.number_input(
                "Body Condition Score (BCS, 1-5):",
                min_value=1.0,
                max_value=5.0,
                value=3.0 if not autofill_btn else 3.25,
                step=0.25,
                help=DMI_FEATURE_METADATA["BCS"]["help"],
            )
            milk_val = st.number_input(
                "Daily Milk Yield (kg/day):",
                min_value=0.0,
                max_value=80.0,
                value=28.5 if not autofill_btn else 32.0,
                step=0.5,
                help=DMI_FEATURE_METADATA["Milk"]["help"],
            )

        with col_f3:
            st.markdown("##### 🥛 Milk Composition")
            pc_val = st.number_input(
                "Protein Content (PC_Milk, g/kg):",
                min_value=15.0,
                max_value=60.0,
                value=32.5 if not autofill_btn else 34.0,
                step=0.5,
                help=DMI_FEATURE_METADATA["PC_Milk"]["help"],
            )
            fat_val = st.number_input(
                "Fat Content (Fat_Milk, g/kg):",
                min_value=20.0,
                max_value=70.0,
                value=38.0 if not autofill_btn else 39.5,
                step=0.5,
                help=DMI_FEATURE_METADATA["Fat_Milk"]["help"],
            )
            lac_val = st.number_input(
                "Lactose Content (Lac_Milk, g/kg):",
                min_value=25.0,
                max_value=65.0,
                value=48.0 if not autofill_btn else 49.0,
                step=0.5,
                help=DMI_FEATURE_METADATA["Lac_Milk"]["help"],
            )

        dmi_notes = st.text_input(
            "Prediction Notes (Optional):",
            placeholder="e.g. Mid-lactation nutritional screening",
        )

        submit_dmi = st.form_submit_button("🔮 Predict Daily Dry Matter Intake", type="primary", use_container_width=True)

    # Handle Prediction
    if submit_dmi:
        feature_dict = {
            "age": age_val,
            "parity": parity_val,
            "DIM": dim_val,
            "BW": bw_val,
            "BCS": bcs_val,
            "Milk": milk_val,
            "PC_Milk": pc_val,
            "Fat_Milk": fat_val,
            "Lac_Milk": lac_val,
        }

        with st.spinner("Processing DMI prediction pipeline..."):
            pred_response = predict_dmi(feature_dict)

            if pred_response["status"] == "error":
                st.error(f"❌ {pred_response['message']}")
                for err in pred_response.get("errors", []):
                    st.markdown(f"- {err}")

            elif pred_response["status"] == "model_required":
                st.info(
                    "### 📋 Input Validation Successful (9/9 Features Valid)\n"
                    f"**Status:** {pred_response['message']}\n\n"
                    "All 9 input features comply with the required schema of the **Pooled Expanded Features Extra Trees Regressor**."
                )

                v_table = pd.DataFrame([
                    {"Feature": DMI_FEATURE_METADATA[k]["label"], "Value": f"{v} {DMI_FEATURE_METADATA[k]['unit']}", "Model Column": k}
                    for k, v in pred_response["validated_inputs"].items()
                ])
                st.table(v_table)

            elif pred_response["status"] == "success":
                st.session_state["pending_dmi_prediction"] = {
                    "predicted_dmi_kg_day": pred_response["predicted_dmi_kg_day"],
                    "timestamp": pred_response["prediction_timestamp"],
                    "input_features": pred_response["input_features"],
                    "notes": dmi_notes,
                }

    # Keep the successful result and save action available after Streamlit reruns.
    pending_dmi_prediction = st.session_state.get("pending_dmi_prediction")
    if pending_dmi_prediction:
        dmi_kg = pending_dmi_prediction["predicted_dmi_kg_day"]
        prediction_timestamp = pending_dmi_prediction["timestamp"]
        st.markdown(
            f"""
            <div style="background-color: #F0FDF4; border: 2px solid #22C55E; padding: 1.5rem; border-radius: 8px; text-align: center; margin-top: 1rem;">
                <h3 style="margin: 0; color: #15803D;">Predicted Daily Dry Matter Intake</h3>
                <h1 style="margin: 0.4rem 0; font-size: 3rem; color: #166534;">{dmi_kg} kg/day</h1>
                <span style="background-color: #DCFCE7; color: #166534; padding: 0.3rem 0.8rem; border-radius: 4px; font-weight: 600; font-size: 0.85rem;">
                    ESTIMATED INTAKE (INRAE DATASET 2)
                </span>
                <p style="margin-top: 0.6rem; color: #4B5563; font-size: 0.9rem;">
                    Estimated physiological consumption. Strictly <b>NOT</b> a prescribed feed ration.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Prediction timestamp: `{prediction_timestamp}`. This is a model estimate, not a measured intake.")
        st.write("**Inputs used:**")
        st.dataframe(pd.DataFrame([
            {"Feature": DMI_FEATURE_METADATA[k]["label"], "Value": f"{v} {DMI_FEATURE_METADATA[k]['unit']}", "Model column": k}
            for k, v in pending_dmi_prediction["input_features"].items()
        ]), hide_index=True, width="stretch")

        if active_animal_id:
            if st.button(f"💾 Save DMI Prediction to Animal {active_animal_id}", type="secondary"):
                try:
                    record_dmi_prediction(
                        animal_id=active_animal_id,
                        predicted_dmi_kg_day=pending_dmi_prediction["predicted_dmi_kg_day"],
                        model_name=DMI_MODEL_NAME,
                        input_features=pending_dmi_prediction["input_features"],
                        notes=pending_dmi_prediction["notes"],
                        timestamp=pending_dmi_prediction["timestamp"],
                    )
                    st.session_state.pop("pending_dmi_prediction", None)
                    st.success(f"✅ DMI prediction saved to Digital Twin **{active_animal_id}**!")
                except Exception as exc:
                    st.error(f"❌ Could not save DMI prediction: {exc}")

    # Previous DMI predictions for selected animal
    if active_animal_id:
        st.divider()
        st.subheader(f"📜 DMI Prediction History for Animal: {active_animal_id}")
        past_dmi = get_dmi_predictions(active_animal_id, limit=25)
        if past_dmi:
            p_df = pd.DataFrame(past_dmi)[["timestamp", "predicted_dmi_kg_day", "model_name", "input_features", "notes"]]
            p_df["input_features"] = p_df["input_features"].apply(lambda row: ", ".join(f"{key}={value}" for key, value in row.items()))
            p_df.columns = ["Timestamp", "Predicted DMI (kg/day)", "Model", "Inputs used", "Notes"]
            st.dataframe(p_df, width="stretch")
        else:
            st.info(f"No DMI predictions recorded yet for animal {active_animal_id}.")


# =========================================================================
# TAB 4: FEED SUPPLY CHAIN
# =========================================================================
with tab_feed:
    st.header("📦 Feed Inventory & Supply Chain")
    st.caption("Track entered inventory, stock coverage, and supplier lead-time risk. No orders are placed automatically.")

    with st.form("feed_inventory_form"):
        st.subheader("Add or update an inventory item")
        feed_name = st.text_input("Feed / ingredient name")
        feed_col1, feed_col2 = st.columns(2)
        with feed_col1:
            stock_kg = st.number_input("Available stock (kg)", min_value=0.0, value=0.0, step=10.0)
            daily_usage_kg = st.number_input("Daily usage (kg/day)", min_value=0.0, value=0.0, step=1.0)
            reorder_threshold_kg = st.number_input("Reorder threshold (kg)", min_value=0.0, value=0.0, step=10.0)
        with feed_col2:
            unit_cost = st.number_input("Unit cost (per kg)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            supplier_name = st.text_input("Supplier name")
            supplier_lead_time_days = st.number_input("Supplier lead time (days)", min_value=0.0, value=0.0, step=1.0)
        st.caption("Saving an existing feed name updates its current stock and supplier details.")
        save_feed_item = st.form_submit_button("Save inventory item", type="primary")

    if save_feed_item:
        try:
            upsert_feed_inventory_item(
                feed_name=feed_name,
                available_stock_kg=stock_kg,
                daily_usage_kg=daily_usage_kg,
                unit_cost=unit_cost,
                supplier_name=supplier_name,
                supplier_lead_time_days=supplier_lead_time_days,
                reorder_threshold_kg=reorder_threshold_kg,
            )
            st.success(f"Inventory item **{feed_name.strip()}** saved.")
        except (TypeError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not save inventory item: {exc}")

    st.subheader("Current inventory")
    inventory = get_feed_inventory()
    if not inventory:
        st.info("No feed inventory entered yet. Add your farm's actual stock values above.")
    else:
        as_of = datetime.now()
        inventory_rows = []
        for item in inventory:
            status = calculate_inventory_status(item, as_of=as_of)
            days_remaining = status["days_of_stock_remaining"]
            stock_out_date = status["estimated_stock_out_date"]
            inventory_rows.append({
                "Feed / Ingredient": item["feed_name"],
                "Available stock (kg)": item["available_stock_kg"],
                "Daily usage (kg/day)": item["daily_usage_kg"],
                "Days of stock remaining": round(days_remaining, 1) if days_remaining is not None else "Not estimated (zero usage)",
                "Estimated stock-out date": stock_out_date.strftime("%Y-%m-%d") if stock_out_date else "—",
                "Reorder status": "Reorder now" if status["reorder_required"] else "Stock above threshold",
                "Supplier lead-time warning": "Stock may run out before delivery" if status["stock_out_before_lead_time"] else "—",
                "Estimated inventory value": status["estimated_inventory_value"],
                "Unit cost (per kg)": item["unit_cost"],
                "Supplier": item["supplier_name"],
                "Lead time (days)": item["supplier_lead_time_days"],
                "Reorder threshold (kg)": item["reorder_threshold_kg"],
                "Last updated": item["last_updated"],
            })
            if status["reorder_required"]:
                st.warning(f"{item['feed_name']}: stock is at or below its reorder threshold.")
            if status["stock_out_before_lead_time"]:
                st.warning(f"{item['feed_name']}: projected stock-out is before the supplier lead time ends.")

        st.dataframe(pd.DataFrame(inventory_rows), hide_index=True, width="stretch")
    
    # ---------------------------------------------------------
    # Herd feed demand and shortage forecast
    # ---------------------------------------------------------
    st.divider()
    st.subheader("🐄 Herd Feed Demand & Shortage Forecast")
    st.write(
        "Estimate ingredient demand from herd size and average daily DMI, "
        "then compare projected usage with the stock and supplier lead time "
        "recorded above. This is a planning estimate, not an automatic purchase order."
    )
    st.warning(
        "For valid calculations, enter inventory stock in **kg of dry matter (DM)**. "
        "If your stock is measured as-fed, convert it to dry-matter kilograms first. "
        "The ration percentages below must total 100%."
    )

    forecast_dmi_source = st.radio(
        "DMI source",
        ["Manual herd average", "Saved Digital Twin predictions"],
        horizontal=True,
        key="forecast_dmi_source",
    )

    herd_daily_dmi = None
    herd_dmi_ready = True
    saved_dmi_values = {}
    dmi_issues = []
    if forecast_dmi_source == "Saved Digital Twin predictions":
        saved_dmi = resolve_saved_herd_dmi(all_animals, get_latest_dmi_prediction)
        saved_dmi_values = saved_dmi["animal_dmi_kg_day"]
        dmi_issues = saved_dmi["issues"]
        herd_dmi_ready = not dmi_issues
        if dmi_issues:
            herd_daily_dmi = None
            st.warning(
                "Saved-prediction mode is blocked until each registered animal has a valid latest DMI prediction: "
                + "; ".join(dmi_issues)
            )
        else:
            herd_daily_dmi = saved_dmi["total_dmi_kg_day"]
            st.info(
                f"Using {len(saved_dmi_values)} registered animals; total herd DMI is "
                f"{herd_daily_dmi:,.2f} kg DM/day."
            )

    forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

    with forecast_col1:
        if forecast_dmi_source == "Manual herd average":
            forecast_herd_count = st.number_input(
                "Animals in herd",
                min_value=1,
                value=10,
                step=1,
                key="forecast_herd_count",
            )
        else:
            forecast_col1.metric("Animals with valid saved DMI", len(saved_dmi_values))

    with forecast_col2:
        if forecast_dmi_source == "Manual herd average":
            forecast_dmi = st.number_input(
                "Average DMI per animal (kg DM/day)",
                min_value=0.1,
                value=20.0,
                step=0.5,
                key="forecast_dmi_per_animal",
            )
            herd_daily_dmi = float(forecast_herd_count) * float(forecast_dmi)
        else:
            forecast_col2.metric(
                "Total herd DMI",
                f"{herd_daily_dmi:,.2f} kg DM/day" if herd_daily_dmi is not None else "Blocked",
            )

    with forecast_col3:
        forecast_days = st.number_input(
            "Planning horizon (days)",
            min_value=1,
            max_value=365,
            value=7,
            step=1,
            key="forecast_days",
        )
    ration_percentages = {}
    total_share_pct = 0.0
    ration_total_valid = False
    forecast_ready = False
    if not inventory:
        st.info("Add at least one ingredient in the inventory form above to build a forecast.")
    else:
        st.markdown("**Ration composition** — enter each ingredient's percentage of total dry matter.")
        share_cols = st.columns(3)
        for idx, item in enumerate(inventory):
            feed_name_key = item["feed_name"]
            with share_cols[idx % len(share_cols)]:
                share_pct = st.number_input(
                    f"{feed_name_key} (%)", min_value=0.0, max_value=100.0,
                    value=0.0, step=5.0, key=f"forecast_share_{feed_name_key}"
                )
            ration_percentages[feed_name_key] = float(share_pct)

        total_share_pct = sum(ration_percentages.values())
        st.caption(f"Ration total: {total_share_pct:.1f}%")
        ration_total_valid = 99.9 <= total_share_pct <= 100.1
        if not ration_total_valid:
            st.info("Enter ration percentages totaling 100% to enable forecasting and procurement.")
        forecast_ready = herd_dmi_ready and ration_total_valid

        if st.button(
            "Calculate feed demand and shortage",
            type="primary",
            key="run_feed_forecast",
            disabled=not forecast_ready,
        ):
            try:
                forecast_demand = calculate_ration_demand(
                    inventory,
                    herd_daily_dmi,
                    ration_percentages,
                    int(forecast_days),
                )
                horizon_demand = sum(row["horizon_demand_kg"] for row in forecast_demand)
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Herd daily DMI", f"{herd_daily_dmi:,.1f} kg DM/day")
                metric2.metric(f"Herd demand ({int(forecast_days)} days)", f"{horizon_demand:,.1f} kg DM")
                metric3.metric("Ingredients tracked", str(len(inventory)))

                risk_names = []
                forecast_rows = []
                for row in forecast_demand:
                    name = row["feed_name"]
                    if row["projected_shortage_kg"] > 0 or row["reorder_required"] or row["lead_time_risk"]:
                        risk_names.append(name)
                    forecast_rows.append({
                        "Ingredient": name,
                        "Ration share (%)": round(row["ration_share_pct"], 1),
                        "Projected use (kg DM/day)": round(row["daily_demand_kg"], 2),
                        f"Demand over {int(forecast_days)} days (kg DM)": round(row["horizon_demand_kg"], 2),
                        "Available stock (kg DM)": round(row["available_stock_kg"], 2),
                        "Projected shortage (kg DM)": round(row["projected_shortage_kg"], 2),
                        "Days of stock at planned ration": round(row["days_of_stock_remaining"], 1) if row["days_of_stock_remaining"] is not None else "Not used",
                        "Reorder required": "Yes" if row["reorder_required"] else "No",
                        "Runs out before lead time": "Yes" if row["lead_time_risk"] else "No",
                        "Supplier": row.get("supplier_name", ""),
                        "Lead time (days)": row["supplier_lead_time_days"],
                    })

                st.dataframe(pd.DataFrame(forecast_rows), hide_index=True, use_container_width=True)
                if risk_names:
                    st.warning("Review these ingredients for shortage/reordering: " + ", ".join(risk_names))
                else:
                    st.success("No projected shortage or reorder risk was detected for this horizon.")
                st.caption(
                    "Forecast assumes constant herd DMI and the entered ration mix throughout the planning horizon. "
                    "It does not place orders or account for delivery quantities."
                )
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"Could not calculate feed demand: {exc}")
                
        st.divider()
        st.subheader("🛒 Procurement Optimization")
        st.caption(
            "Prioritizes feed purchases using forecast demand, supplier lead time, "
            "safety stock, and your available budget."
        )

        proc_col1, proc_col2, proc_col3 = st.columns(3)

        with proc_col1:
            proc_horizon = st.number_input(
                "Procurement planning horizon (days)",
                min_value=1,
                value=7,
                step=1,
                key="proc_horizon_days",
            )

        with proc_col2:
            proc_safety_days = st.number_input(
                "Safety stock (days)",
                min_value=0,
                value=3,
                step=1,
                key="proc_safety_stock_days",
            )

        with proc_col3:
            proc_budget = st.number_input(
                "Procurement budget",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                key="proc_budget",
            )

        if st.button(
            "Optimize Feed Procurement",
            type="primary",
            key="run_procurement_optimizer",
            disabled=not forecast_ready,
        ):
            try:
                procurement_demand = calculate_ration_demand(
                    inventory,
                    herd_daily_dmi,
                    ration_percentages,
                    int(proc_horizon),
                )
                # Use the exact ingredient daily demand derived from the ration.
                procurement_inventory = []
                for row in procurement_demand:
                    procurement_inventory.append({
                        **row,
                        "daily_usage_kg": row["daily_demand_kg"],
                    })

                result = optimize_procurement(
                    procurement_inventory,
                    planning_horizon_days=int(proc_horizon),
                    safety_stock_days=int(proc_safety_days),
                    budget=float(proc_budget),
                )

                total_required = result["total_required_cost"]
                total_allocated = result["total_allocated_cost"]

                m1, m2, m3 = st.columns(3)
                m1.metric("Cost of full requirements", f"{total_required:,.2f}")
                m2.metric("Planned purchase cost", f"{total_allocated:,.2f}")
                m3.metric("Unallocated requirement", f"{max(0, total_required - total_allocated):,.2f}")

                if total_required > proc_budget:
                    st.warning(
                        "The budget does not cover all calculated requirements. "
                        "Review the priority and unmet quantities before purchasing."
                    )
                else:
                    st.success("The available budget covers all calculated purchase requirements.")

                recommendations = result["recommendations"]

                if recommendations:
                    proc_df = pd.DataFrame(recommendations)

                    display_columns = [
                        "feed_name",
                        "priority",
                        "days_of_stock_remaining",
                        "target_stock_kg",
                        "required_order_kg",
                        "recommended_order_kg",
                        "unit_cost",
                        "estimated_purchase_cost",
                        "supplier_name",
                        "supplier_lead_time_days",
                        "suggested_order_date",
                        "budget_limited",
                    ]
                    display_columns = [col for col in display_columns if col in proc_df.columns]
                    proc_df = proc_df[display_columns].rename(columns={
                        "feed_name": "Ingredient",
                        "priority": "Priority",
                        "days_of_stock_remaining": "Stock remaining (days)",
                        "target_stock_kg": "Target stock (kg DM)",
                        "required_order_kg": "Required order (kg DM)",
                        "recommended_order_kg": "Recommended order (kg DM)",
                        "unit_cost": "Unit cost",
                        "estimated_purchase_cost": "Estimated cost",
                        "supplier_name": "Supplier",
                        "supplier_lead_time_days": "Lead time (days)",
                        "suggested_order_date": "Suggested order date",
                        "budget_limited": "Budget limited",
                    })
                    st.dataframe(proc_df, hide_index=True, use_container_width=True)

                    limited_items = [r for r in recommendations if r.get("budget_limited", False)]
                    if limited_items:
                        st.warning("Some orders are only partially funded.")
                        st.dataframe(
                            pd.DataFrame([
                                {
                                    "Ingredient": r["feed_name"],
                                    "Required (kg DM)": r["required_order_kg"],
                                    "Recommended (kg DM)": r["recommended_order_kg"],
                                    "Unfunded (kg DM)": round(r["required_order_kg"] - r["recommended_order_kg"], 2),
                                    "Reason": r.get("budget_note", ""),
                                }
                                for r in limited_items
                            ]),
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info("No purchase requirements were left partially funded.")
                else:
                    st.success("No additional feed purchases are required for the selected settings.")

                st.caption(
                    "Recommendations are decision support, not placed orders. Quantities assume inventory and "
                    "forecast demand use the same units (kg dry matter). Verify stock, prices, supplier availability, "
                    "and delivery dates before purchasing."
                )
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"Could not optimize feed procurement: {exc}")



# =========================================================================
# TAB 5: HEALTH HISTORY & DASHBOARD
# =========================================================================
with tab_history:
    st.header("📈 Health History & Telemetry Dashboard")

    if not active_animal_id:
        st.info("👆 Please select an active animal profile in the sidebar to review its longitudinal history.")
    else:
        st.subheader(f"Historical Audit Trail for Animal: {active_animal_id}")

        obs_list = get_observations(active_animal_id, limit=100)
        sensor_list = get_sensor_readings(active_animal_id, limit=100)

        # Observations History
        st.write("#### 🖼️ Computer-Vision Disease Screenings")
        if obs_list:
            obs_df = pd.DataFrame(obs_list)[["timestamp", "cv_prediction", "cv_confidence", "model_used", "notes"]]
            obs_df["cv_confidence"] = obs_df["cv_confidence"].apply(lambda v: f"{v*100:.1f}%" if v <= 1.0 else f"{v:.1f}%")
            obs_df.columns = ["Timestamp", "Predicted Condition", "Confidence", "Model", "Notes"]
            st.dataframe(obs_df, width="stretch")
        else:
            st.info("No vision screening observations logged yet.")

        st.divider()

        # Sensor & Telemetry History
        st.write("#### 🌡️ Vital Signs & Telemetry History")
        if sensor_list:
            s_df = pd.DataFrame(sensor_list)[
                ["timestamp", "body_temperature_c", "activity_level", "feed_intake",
                 "ambient_temperature_c", "humidity_percent", "data_source", "is_simulated"]
            ]
            s_df["Provenance"] = s_df["is_simulated"].apply(lambda x: "SIMULATED" if x else "MANUAL")
            s_df = s_df.drop(columns=["is_simulated"])
            s_df.columns = ["Timestamp", "Body Temp (°C)", "Activity", "Feed Intake", "Ambient Temp (°C)", "Humidity (%)", "Source", "Type"]
            st.dataframe(s_df, width="stretch")

            # Chart Body Temperature Trajectory
            if "Body Temp (°C)" in s_df.columns:
                st.write("**Body Temperature Trend (°C):**")
                chart_df = s_df[["Timestamp", "Body Temp (°C)"]].dropna()
                chart_df = chart_df.set_index("Timestamp")
                st.line_chart(chart_df, y="Body Temp (°C)", color="#DC2626")
        else:
            st.info("No sensor telemetry logged yet.")

        st.divider()

        # DMI Prediction History
        st.write("#### 🌾 Dry Matter Intake (DMI) Prediction History")
        dmi_records = get_dmi_predictions(active_animal_id, limit=100)
        if dmi_records:
            dmi_df = pd.DataFrame(dmi_records)[["timestamp", "predicted_dmi_kg_day", "model_name", "notes"]]
            dmi_df.columns = ["Timestamp", "Predicted DMI (kg/day)", "Model", "Notes"]
            st.dataframe(dmi_df, width="stretch")

            st.write("**DMI Trajectory (kg/day):**")
            dmi_chart_df = dmi_df[["Timestamp", "Predicted DMI (kg/day)"]].set_index("Timestamp")
            st.line_chart(dmi_chart_df, y="Predicted DMI (kg/day)", color="#16A34A")
        else:
            st.info("No DMI predictions recorded yet for this animal.")


# =========================================================================
# TAB 6: MODEL COMPARISON & BENCHMARKS (Preserved)
# =========================================================================
with tab_benchmarks:
    st.header("📊 Model Architecture Evaluation & Selection")
    st.write(
        "Below is the empirical evaluation of all models tested during experimentation on the untouched test dataset (487 test images). "
        "Each architecture was trained to classify cattle images into **Healthy**, **Foot-and-Mouth Disease**, and **Lumpy Skin Disease**."
    )

    benchmark_data = [
        {
            "Architecture": "DenseNet-121",
            "Weights Location": "models/densenet121_cattle_best.pth",
            "Test Accuracy": "96.30%",
            "Macro F1": "0.9639",
            "FMD F1": "0.9683",
            "LSD F1": "0.9667",
            "Healthy F1": "0.9567",
            "Grad-CAM Supported": "Yes (features.denseblock4.denselayer16.conv2)",
            "Local Status": "✅ Fully Available & Loaded (Recommended)",
        },
        {
            "Architecture": "ResNet-50",
            "Weights Location": "/kaggle/working/resnet50_cattle_best.pth",
            "Test Accuracy": "95.89%",
            "Macro F1": "0.9637",
            "FMD F1": "0.9955",
            "LSD F1": "0.9438",
            "Healthy F1": "0.9519",
            "Grad-CAM Supported": "Yes (layer4[-1])",
            "Local Status": "⚠️ Trained on Kaggle; weights not saved locally",
        },
        {
            "Architecture": "EfficientNet-B0 (Torchvision)",
            "Weights Location": "models/efficientnet_cattle_best.pth",
            "Test Accuracy": "94.66%",
            "Macro F1": "0.9525",
            "FMD F1": "0.9911",
            "LSD F1": "0.9300",
            "Healthy F1": "0.9364",
            "Grad-CAM Supported": "Yes (features[-1])",
            "Local Status": "✅ Fully Available & Loaded",
        },
        {
            "Architecture": "DINOv2 ViT-S/14",
            "Weights Location": "models/dinov2_vits14_cattle_best.pth",
            "Test Accuracy": "94.05%",
            "Macro F1": "0.9444",
            "FMD F1": "0.9683",
            "LSD F1": "0.9359",
            "Healthy F1": "0.9289",
            "Grad-CAM Supported": "Attention Rollout",
            "Local Status": "⚠️ Requires torch.hub remote download",
        },
        {
            "Architecture": "EfficientNet-B0 (LMO Optimized)",
            "Weights Location": "models/efficientnet_b0_lmo_best.pth",
            "Test Accuracy": "93.02%",
            "Macro F1": "0.9354",
            "FMD F1": "0.9689",
            "LSD F1": "0.9213",
            "Healthy F1": "0.9160",
            "Grad-CAM Supported": "No",
            "Local Status": "⚠️ Requires external timm library",
        },
    ]

    bench_df = pd.DataFrame(benchmark_data)
    st.dataframe(bench_df, width="stretch")

    st.subheader("💡 Why DenseNet-121 Was Selected as Default:")
    st.markdown(
        """
        1. **Highest Overall Accuracy & F1-Score:** DenseNet-121 achieved **96.30% test accuracy** and **0.9639 macro F1-score**, beating all other architectures including ResNet-50 (95.89%) and EfficientNet-B0 (94.66%).
        2. **Dense Feature Reuse:** Dense connectivity allows each convolutional layer to receive collective feature representations from all preceding layers, making it particularly effective at detecting subtle skin nodules and oral mucosal lesions.
        3. **Self-Contained Offline Weights:** The trained weights file `models/densenet121_cattle_best.pth` (28.4 MB) is fully present locally in the repository and runs directly on standard PyTorch / Torchvision without requiring external hub or timm downloads.
        4. **Native Grad-CAM Explainability:** Supports Grad-CAM visual heatmaps via `model.features.denseblock4.denselayer16.conv2`.
        """
    )


# =========================================================================
# TAB 7: CLINICAL DISEASE GUIDE (Preserved)
# =========================================================================
with tab_guide:
    st.header("📚 Clinical Guide: Cattle Diseases")

    st.subheader("1. Healthy Cattle (Control Class)")
    st.markdown(
        """
        - **General Appearance:** Bright, alert, active posture with smooth, clean skin and coat.
        - **Muzzle & Oral Cavity:** Moist, clean muzzle with absence of ulcers, erosions, blisters, or excessive drooling.
        - **Locomotion:** Normal gait without limping, reluctance to move, or foot lesions.
        """
    )

    st.subheader("2. Foot-and-Mouth Disease (FMD)")
    st.markdown(
        """
        - **Etiology:** Aphthovirus of the family *Picornaviridae*. Highly contagious transboundary animal disease.
        - **Primary Signs:** High fever (40–41°C / 104–106°F), severe depression, loss of appetite, and sudden drop in milk yield.
        - **Characteristic Lesions:**
          - Vesicles (blisters) and raw erosions on the tongue, dental pad, gums, inner lips, and muzzle.
          - Blisters between hooves (interdigital cleft) and on the coronary band, causing severe lameness.
          - Copious, frothy, or ropy salivation ("stringy drool") and smacking of lips.
        """
    )

    st.subheader("3. Lumpy Skin Disease (LSD)")
    st.markdown(
        """
        - **Etiology:** Capripoxvirus of the family *Poxviridae*. Transmitted primarily by blood-feeding arthropod vectors (mosquitoes, stable flies, ticks).
        - **Primary Signs:** Acute fever followed 1–2 days later by characteristic skin eruptions.
        - **Characteristic Lesions:**
          - Circumscribed, firm, round, raised skin nodules (2–5 cm in diameter) across the entire body, especially head, neck, perineum, udder, and limbs.
          - Deep dermal necrosis causing "sit-fast" lesions that slough off and leave open wounds prone to secondary infection.
          - Enlarged superficial lymph nodes (prescapular and precrural) and edema in dependent parts (dewlap, legs).
        """
    )
