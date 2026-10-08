import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

# Setup Project Base Directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import Local Modules
from part1_return_risk.predict import predict_return_risk
from part2_image_classifier.predict import predict_image
from part3_support_agent.agent import generate_support_response

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OrderSense-AI | E-Commerce Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Professional Light CSS Styling for OrderSense-AI
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Clean Light Theme Custom Styling */
    .main {
        background-color: #f8fafc;
        padding-top: 1rem;
    }
    
    .stAppHeader {
        background-color: #f8fafc;
    }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: #ffffff;
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2);
    }
    .header-banner h1 {
        font-family: 'Inter', system-ui, sans-serif;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-banner h1 span {
        color: #facc15;
    }
    .header-banner p {
        font-size: 14px;
        color: #e0e7ff;
        margin: 6px 0 0 0;
    }

    /* Card Panels */
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .risk-high {
        background-color: #fee2e2;
        color: #dc2626;
        border: 1px solid #fca5a5;
    }
    .risk-medium {
        background-color: #fef3c7;
        color: #d97706;
        border: 1px solid #fcd34d;
    }
    .risk-low {
        background-color: #dcfce7;
        color: #16a34a;
        border: 1px solid #86efac;
    }

    /* AI Response Container */
    .agent-response-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        font-size: 14.5px;
        line-height: 1.75;
        color: #1e293b;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="header-banner">
        <h1>⚡ OrderSense-AI Platform</h1>
        <p>Multi-Modal E-Commerce Order Return Intelligence & AI Support System</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Control Panel & Preset Scenarios
# -----------------------------------------------------------------------------
st.sidebar.title("📌 Control Panel")
st.sidebar.markdown("Select a quick test scenario or enter details manually.")

# Preset Scenarios State Initialization
if "preset_loaded" not in st.session_state:
    st.session_state.preset_loaded = "high"

col_s1, col_s2, col_s3 = st.sidebar.columns(3)
if col_s1.button("🔥 High", help="Load High Risk Preset"):
    st.session_state.preset_loaded = "high"
if col_s2.button("🟠 Med", help="Load Medium Risk Preset"):
    st.session_state.preset_loaded = "medium"
if col_s3.button("🟢 Low", help="Load Low Risk Preset"):
    st.session_state.preset_loaded = "low"

# Load Preset Default Values
preset = st.session_state.preset_loaded
if preset == "high":
    def_cat = "Footwear"
    def_price = 3499.0
    def_disc = 45.0
    def_pay = "COD"
    def_tenure = 120
    def_orders = 4
    def_returns = 3
    def_rating = 2.0
    def_query = "The shoe size is too tight. How do I request an exchange or return?"
elif preset == "medium":
    def_cat = "Apparel"
    def_price = 1499.0
    def_disc = 30.0
    def_pay = "Wallet"
    def_tenure = 250
    def_orders = 7
    def_returns = 1
    def_rating = 4.0
    def_query = "The product color looks slightly different from the image. Can I return it?"
else:
    def_cat = "Electronics"
    def_price = 12999.0
    def_disc = 10.0
    def_pay = "Prepaid_UPI"
    def_tenure = 850
    def_orders = 28
    def_returns = 0
    def_rating = 5.0
    def_query = "How do I register the brand warranty and download my official tax invoice?"

st.sidebar.divider()
st.sidebar.markdown("### 📊 System Metadata")
st.sidebar.info(
    "**Architecture:** Multi-Modal ML Pipeline\n\n"
    "• **Tabular ML:** RandomForestClassifier\n"
    "• **Vision Model:** TensorFlow Lite (224×224)\n"
    "• **AI Support Agent:** Mistral AI API / Local Engine\n\n"
    "**Project:** OrderSense-AI"
)

# -----------------------------------------------------------------------------
# Main Application Grid Layout
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9], gap="medium")

with col_left:
    st.markdown("### 📦 Order Details & Telemetry")
    
    with st.expander("1. Product & Payment Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            product_category = st.selectbox(
                "Product Category",
                ["Footwear", "Apparel", "Electronics", "Home", "Beauty"],
                index=["Footwear", "Apparel", "Electronics", "Home", "Beauty"].index(def_cat)
            )
            price_inr = st.number_input("Price (₹)", min_value=1.0, value=def_price, step=100.0)
        with col2:
            discount_pct = st.number_input("Discount (%)", min_value=0.0, max_value=90.0, value=def_disc, step=5.0)
            payment_method = st.selectbox(
                "Payment Method",
                ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"],
                index=["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"].index(def_pay)
            )

    with st.expander("2. Customer History & Logistics", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            customer_tenure_days = st.number_input("Customer Tenure (Days)", min_value=1, value=def_tenure)
            num_previous_orders = st.number_input("Previous Orders", min_value=0, value=def_orders)
        with col4:
            num_previous_returns = st.number_input("Previous Returns", min_value=0, value=def_returns)
            rating_given = st.number_input("Rating Given (Optional)", min_value=1.0, max_value=5.0, value=def_rating)

    with st.expander("3. Product Photo Upload (TFLite Classifier)", expanded=True):
        uploaded_image = st.file_uploader(
            "Upload Product Image for Vision Classification",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, PNG, WEBP"
        )
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Product Photo", width=180)

    with st.expander("4. Customer Inquiry Query", expanded=True):
        question = st.text_area(
            "Customer Question / Message",
            value=def_query,
            height=90,
            placeholder="Type customer question here..."
        )
        
        # Quick Query Chips
        st.caption("Quick Queries:")
        chip_col1, chip_col2, chip_col3 = st.columns(3)
        if chip_col1.button("Size Issue"):
            question = "The size is too tight. How do I request an exchange or return?"
        if chip_col2.button("Return Policy"):
            question = "What is the return and replacement policy window for this product?"
        if chip_col3.button("Damaged Item"):
            question = "The product arrived damaged. How can I get a priority replacement?"

    analyze_btn = st.button("⚡ Analyze Order & Consult AI Support", type="primary", use_container_width=True)

with col_right:
    st.markdown("### 🤖 OrderSense-AI Support Output")

    if analyze_btn or "last_analysis" in st.session_state:
        if analyze_btn:
            with st.spinner("Processing OrderSense-AI Intelligence Pipeline..."):
                # 1. Vision Classifier Inference
                detected_category = product_category
                image_confidence = 0.95
                if uploaded_image is not None:
                    try:
                        image_bytes = uploaded_image.getvalue()
                        vision_res = predict_image(image_bytes)
                        detected_category = vision_res["category"]
                        image_confidence = float(vision_res["confidence"])
                    except Exception as img_err:
                        st.warning(f"Vision Classifier Notice: {img_err}")

                # 2. Return Risk ML Inference
                order_data = {
                    "product_category": product_category,
                    "price_inr": float(price_inr),
                    "discount_pct": float(discount_pct),
                    "payment_method": payment_method,
                    "customer_tenure_days": int(customer_tenure_days),
                    "num_previous_orders": int(num_previous_orders),
                    "num_previous_returns": int(num_previous_returns),
                    "delivery_distance_km": 150.0,
                    "delivery_days": 4,
                    "is_weekend_order": 0,
                    "rating_given": float(rating_given) if rating_given else None
                }
                
                risk_res = predict_return_risk(order_data)
                return_risk = risk_res["risk_level"]
                return_probability = float(risk_res["probability"])

                # 3. AI Agent Response
                agent_response = generate_support_response(
                    question=question,
                    order_data=order_data,
                    return_risk=return_risk,
                    return_probability=return_probability,
                    product_category=detected_category,
                    image_confidence=image_confidence
                )

                # Save Analysis to State
                st.session_state.last_analysis = {
                    "agent_response": agent_response,
                    "return_risk": return_risk,
                    "return_probability": return_probability,
                    "detected_category": detected_category,
                    "image_confidence": image_confidence
                }

        # Retrieve & Render Results
        res_data = st.session_state.last_analysis
        
        # Render AI Agent Response Box
        st.markdown(f"""
            <div class="agent-response-box">
                <div style="font-weight: 700; font-size: 15px; margin-bottom: 10px; color: #1e3a8a; display: flex; align-items: center; gap: 8px;">
                    🤖 OrderSense-AI Support Specialist Guidance
                </div>
                <div>{res_data['agent_response']}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Real-Time ML Telemetry")

        # Telemetry Metrics Grid
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.caption("ML RETURN RISK (RandomForest)")
            risk_tier = res_data["return_risk"]
            risk_class = f"risk-{risk_tier.lower()}"
            
            st.markdown(f'<span class="risk-badge {risk_class}">{risk_tier} RISK</span>', unsafe_allow_html=True)
            st.metric(label="Return Probability", value=f"{res_data['return_probability'] * 100:.1f}%")
            st.progress(min(1.0, max(0.0, res_data["return_probability"])))
            st.markdown('</div>', unsafe_allow_html=True)

        with m_col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.caption("VISION CLASSIFIER (TFLite)")
            st.markdown(f"**{res_data['detected_category']}**")
            st.metric(label="Model Confidence", value=f"{res_data['image_confidence'] * 100:.1f}%")
            st.progress(min(1.0, max(0.0, res_data["image_confidence"])))
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Enter order details on the left and click **Analyze Order & Consult AI Support** to generate real-time intelligence.")
