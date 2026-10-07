import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="OrderSense AI",
    page_icon="🛒",
    layout="wide"
)

model_data = joblib.load(
    "part1_return_prediction/models/return_prediction_model.pkl"
)

model = model_data["model"]
threshold = model_data["threshold"]

PRODUCT_CATEGORIES = [
    "Apparel",
    "Electronics",
    "Footwear",
    "Home"
]

PAYMENT_METHODS = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash on Delivery"
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Light Theme Main Canvas */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Compact Main Container Width */
    .main .block-container {
        max-width: 750px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
    }

    /* Main Title & Subtitle Styling */
    .main-title {
        text-align: center;
        font-size: 50px;
        font-weight: 820;
        color: #1e3a8a;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        font-weight: 510;
        color: #475569;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 12px;
        margin-bottom: 12px;
        padding-left: 10px;
        border-left: 4px solid #2563eb;
    }

    /* Compact Input Fields & Select Boxes */
    input, select, textarea,
    [data-baseweb="input"], [data-baseweb="select"], [data-baseweb="base-input"],
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
    .stSelectbox div[data-baseweb="select"], .stNumberInput input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 4px 10px !important;
        min-height: 38px !important;
        box-shadow: 0 3px 10px rgba(148, 163, 184, 0.08) !important;
        transition: all 0.2s ease !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    /* Focus & Hover effects on input fields */
    div[data-baseweb="select"]:hover > div, div[data-baseweb="input"]:hover > div {
        border-color: #94a3b8 !important;
        box-shadow: 0 5px 14px rgba(148, 163, 184, 0.12) !important;
    }

    div[data-baseweb="select"]:focus-within > div, div[data-baseweb="input"]:focus-within > div,
    input:focus, select:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    /* Input Field Labels */
    label[data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        margin-bottom: 4px !important;
    }

    /* Input SVG Icons & Arrows */
    div[data-baseweb="select"] svg, div[data-baseweb="input"] svg, [data-baseweb="icon"] svg {
        fill: #475569 !important;
        color: #475569 !important;
    }

    /* Number Input Increment/Decrement Buttons (+ / -) */
    [data-testid="stNumberInput"] button, [data-baseweb="input"] button {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }

    [data-testid="stNumberInput"] button:hover, [data-baseweb="input"] button:hover {
        background-color: #e2e8f0 !important;
        color: #1e293b !important;
    }

    /* Selectbox Dropdown Menu Popover */
    div[data-baseweb="popover"], div[data-baseweb="popover"] *,
    ul[role="listbox"], li[role="option"], [data-baseweb="menu"], [data-baseweb="menu"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #e2e8f0 !important;
        color: #1e293b !important;
    }

    /* Small/Compact Predict Button Styling & Shadow */
    div.stButton {
        display: flex !important;
        justify-content: center !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 24px !important;
        width: 250px !important;
        max-width: 250px !important;
        margin: 0 auto !important;
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.28) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 7px 20px rgba(37, 99, 235, 0.38) !important;
        transform: translateY(-1px) !important;
    }

    /* Metric Result Cards */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        padding: 16px 12px !important;
        border-radius: 12px !important;
        box-shadow: 0 5px 15px rgba(148, 163, 184, 0.1), 0 2px 5px rgba(0, 0, 0, 0.03) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        text-align: center !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.14) !important;
        border-color: #93c5fd !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 4px !important;
    }

    [data-testid="stMetricValue"], [data-testid="stMetricValue"] *, div[data-testid="stMetricValue"] > div {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 28px !important;
        line-height: 1.2 !important;
    }

    /* Progress Bar */
    [data-testid="stProgress"] > div > div {
        background-color: #e2e8f0 !important;
        border-radius: 8px !important;
        height: 10px !important;
    }

    [data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        border-radius: 8px !important;
    }

    /* Notification Alerts Shadow */
    div[data-testid="stNotification"], .stAlert {
        border-radius: 10px !important;
        padding: 14px 18px !important;
        box-shadow: 0 5px 15px rgba(148, 163, 184, 0.1) !important;
        font-weight: 600 !important;
    }

    /* Dataframe / Info Summary Table Card & Shadow */
    [data-testid="stDataFrame"], .stDataFrame, div[data-testid="stTable"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 6px !important;
        box-shadow: 0 5px 15px rgba(148, 163, 184, 0.1) !important;
    }

    /* Footer Container & Shadow */
    .footer {
        text-align: center;
        font-size: 12.5px;
        color: #64748b;
        margin-top: 30px;
        padding: 14px 20px;
        background-color: #00ddef;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(148, 163, 184, 0.06) !important;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🛒 OrderSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based E-commerce Return Prediction System</div>',
    unsafe_allow_html=True
)

st.divider()

st.markdown(
    '<div class="section-title">📦 Order Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    product_category = st.selectbox(
        "Product Category",
        PRODUCT_CATEGORIES
    )

    price_inr = st.number_input(
        "Price (₹)",
        min_value=0.0,
        max_value=100000.0,
        value=2500.0,
        step=100.0
    )

    discount_pct = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    payment_method = st.selectbox(
        "Payment Method",
        PAYMENT_METHODS
    )

    customer_tenure_days = st.number_input(
        "Customer Tenure (Days)",
        min_value=0,
        max_value=10000,
        value=300,
        step=1
    )

with col2:

    num_previous_orders = st.number_input(
        "Previous Orders",
        min_value=0,
        max_value=500,
        value=10,
        step=1
    )

    num_previous_returns = st.number_input(
        "Previous Returns",
        min_value=0,
        max_value=500,
        value=2,
        step=1
    )

    delivery_distance_km = st.number_input(
        "Delivery Distance (km)",
        min_value=0.0,
        max_value=2000.0,
        value=15.0,
        step=1.0
    )

    delivery_days = st.number_input(
        "Delivery Days",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )

    is_weekend_order = st.selectbox(
        "Weekend Order?",
        ["Yes", "No"]
    )

st.divider()

if st.button(
    "🔮 Predict Return Risk",
    use_container_width=True
):

    errors = []

    if product_category not in PRODUCT_CATEGORIES:
        errors.append("Invalid product category.")

    if price_inr < 153 or price_inr > 44950:
        errors.append(
            "Price must be between ₹153 and ₹44,950."
        )

    if discount_pct < 0 or discount_pct > 75:
        errors.append(
            "Discount must be between 0% and 75%."
        )

    if payment_method not in PAYMENT_METHODS:
        errors.append("Invalid payment method.")

    if customer_tenure_days < 0 or customer_tenure_days > 5000:
        errors.append(
            "Customer tenure must be between 0 and 5000 days."
        )

    if num_previous_orders < 0 or num_previous_orders > 100:
        errors.append(
            "Previous orders must be between 0 and 100."
        )

    if num_previous_returns < 0 or num_previous_returns > 100:
        errors.append(
            "Previous returns must be between 0 and 100."
        )

    if num_previous_returns > num_previous_orders:
        errors.append(
            "Previous returns cannot be greater than previous orders."
        )

    if delivery_distance_km < 0 or delivery_distance_km > 500:
        errors.append(
            "Delivery distance must be between 0 and 500 km."
        )

    if delivery_days < 1 or delivery_days > 12:
        errors.append(
            "Delivery days must be between 1 and 12 days."
        )

    if is_weekend_order not in ["Yes", "No"]:
        errors.append("Invalid weekend order value.")

    if errors:

        st.error("Please correct the following inputs:")

        for error in errors:
            st.write(f"• {error}")

        st.stop()

    weekend_value = 1 if is_weekend_order == "Yes" else 0

    new_order = pd.DataFrame([
        {
            "product_category": product_category,
            "price_inr": price_inr,
            "discount_pct": discount_pct,
            "payment_method": payment_method,
            "customer_tenure_days": customer_tenure_days,
            "num_previous_orders": num_previous_orders,
            "num_previous_returns": num_previous_returns,
            "delivery_distance_km": delivery_distance_km,
            "delivery_days": delivery_days,
            "is_weekend_order": weekend_value
        }
    ])

    new_order["previous_return_rate"] = (
        new_order["num_previous_returns"]
        / new_order["num_previous_orders"].replace(0, 1)
    )

    probability = model.predict_proba(new_order)[0][1]

    prediction = int(probability >= threshold)

    probability_percent = probability * 100

    st.markdown("## 📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Return Probability",
            f"{probability_percent:.2f}%"
        )

    with result_col2:
        st.metric(
            "Model Threshold",
            f"{threshold:.2f}"
        )

    with result_col3:
        if probability_percent >= 60:
            risk_level = "High Risk"
        elif probability_percent >= 40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        st.metric(
            "Risk Level",
            risk_level
        )

    st.progress(float(probability))

    if prediction == 1:
        st.error("🔴 Prediction: Returned")
    else:
        st.success("🟢 Prediction: Not Returned")

    st.markdown("### 🔎 Prediction Explanation")

    if prediction == 1:
        st.write(
            f"""
            The model estimated a return probability of
            **{probability_percent:.2f}%**.

            The probability is above the selected model threshold
            of **{threshold:.2f}**, so the system predicts that this
            order is likely to be **returned**.
            """
        )
    else:
        st.write(
            f"""
            The model estimated a return probability of
            **{probability_percent:.2f}%**.

            The probability is below the selected model threshold
            of **{threshold:.2f}**, so the system predicts that this
            order is likely to be **not returned**.
            """
        )

    st.markdown("### 📋 Order Summary")

    summary = pd.DataFrame(
        {
            "Feature": [
                "Product Category",
                "Price",
                "Discount",
                "Payment Method",
                "Customer Tenure",
                "Previous Orders",
                "Previous Returns",
                "Previous Return Rate",
                "Delivery Distance",
                "Delivery Days",
                "Weekend Order"
            ],
            "Value": [
                product_category,
                f"₹{price_inr:,.2f}",
                f"{discount_pct:.1f}%",
                payment_method,
                f"{customer_tenure_days} days",
                num_previous_orders,
                num_previous_returns,
                f"{new_order['previous_return_rate'].iloc[0]:.2f}",
                f"{delivery_distance_km:.1f} km",
                delivery_days,
                is_weekend_order
            ]
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.markdown(
    """
    <div class="footer">
    <b>Disclaimer:</b> OrderSense AI is an educational machine
    learning project. Predictions are model-based estimates and
    should not be considered guaranteed outcomes.
    </div>
    """,
    unsafe_allow_html=True
)
