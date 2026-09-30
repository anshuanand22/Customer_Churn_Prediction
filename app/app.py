import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "models/churn_model.pkl"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN APPLICATION
    ===================================================== */

    .stApp {
        background-color: #1F2937;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       MAIN TITLE
    ===================================================== */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #9AA35A;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 16px;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 25px;
    }


    /* =====================================================
       SECTION HEADERS
    ===================================================== */

    .section-title {
        background: #1e293b;
        color: #ffffff;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 19px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 18px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.20);
    }


    /* =====================================================
       INPUT LABELS
    ===================================================== */

    .stSelectbox label,
    .stNumberInput label {
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #cbd5e1 !important;
    }


    /* =====================================================
       INPUT BOXES
    ===================================================== */

    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #ffffff !important;
        background-color: #2563EB !important;
    }

    .stNumberInput input {
        border-radius: 8px !important;
        border: 1px solid #ffffff !important;
        background-color: #2563EB !important;
        color: #ffffff !important;
    }


    /* Selectbox text */
    .stSelectbox div[data-baseweb="select"] span {
        color: #ffffff !important;
    }


    /* =====================================================
       PREDICTION BUTTON
    ===================================================== */

    .stButton > button {
        width: 100%;

        background: linear-gradient(
            90deg,
            #b71c1c,
            #e53935
        );

        color: white;

        border: none;
        border-radius: 10px;

        padding: 15px 20px;

        font-size: 22px;
        font-weight: 700;

        box-shadow:
            0 5px 15px rgba(183, 28, 28, 0.30);

        transition: all 0.3s ease;
    }

    .stButton > button:hover {

        background: linear-gradient(
            90deg,
            #8e0000,
            #c62828
        );

        transform: translateY(-2px);

        box-shadow:
            0 8px 20px rgba(183, 28, 28, 0.40);
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .sidebar-logo {
        font-size: 27px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #cbd5e1 !important;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    .sidebar-card {
        background: #1e293b;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        border: 1px solid #334155;
    }


    /* =====================================================
       PREDICTION METRIC CARDS
    ===================================================== */

    [data-testid="stMetric"] {
        background: #1e293b !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid #334155 !important;

        box-shadow:
            0 4px 15px rgba(0,0,0,0.25);
    }


    /* Metric label */

    [data-testid="stMetricLabel"] {
        color: #2dd4bf  !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }


    /* Metric value */

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }


    /* =====================================================
       PROBABILITY SECTION
    ===================================================== */

    .stProgress {
        background-color: #334155 !important;
    }


    /* Probability heading */

    h3 {
        color: #f8fafc !important;
    }


    /* Probability caption */

    [data-testid="stCaptionContainer"] {
        color: #cbd5e1 !important;
    }


    /* =====================================================
       DATAFRAME
    ===================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #334155;
    }


    /* =====================================================
       DIVIDER
    ===================================================== */

    hr {
        border-color: #334155 !important;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;

        font-size: 13px;

        margin-top: 45px;
        padding-top: 20px;

        border-top: 1px solid #334155;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">
            📊 Churn AI
        </div>

        <div class="sidebar-subtitle">
            Customer Churn Prediction System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### About")

    st.write(
        "This application uses machine learning "
        "to predict whether a customer is likely "
        "to leave the service."
    )

    st.markdown(
        """
        <div class="sidebar-card">

        <b>System Features</b>

        <br><br>

        • Customer information<br>
        • Service details<br>
        • Contract information<br>
        • Financial information<br>
        • Churn probability<br>
        • Machine learning prediction

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption(
        "Python • Scikit-learn • Streamlit"
    )


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    """
    <div class="main-title">
        Customer Churn Prediction
    </div>

    <div class="subtitle">
        AI-powered customer retention and churn analysis system
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        👤 Customer Information
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [
            0,
            1
        ]
    )

    partner = st.selectbox(
        "Partner",
        [
            "Yes",
            "No"
        ]
    )

    dependents = st.selectbox(
        "Dependents",
        [
            "Yes",
            "No"
        ]
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    phone_service = st.selectbox(
        "Phone Service",
        [
            "Yes",
            "No"
        ]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


# =========================================================
# COLUMN 3
# =========================================================

with col3:

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# =========================================================
# SERVICE INFORMATION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📡 Service Information
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col2:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )


with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        💰 Financial Information
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


# =========================================================
# CREATE CUSTOMER DATAFRAME
# =========================================================

customer_data = pd.DataFrame({

    "gender": [gender],

    "SeniorCitizen": [senior_citizen],

    "Partner": [partner],

    "Dependents": [dependents],

    "tenure": [tenure],

    "PhoneService": [phone_service],

    "MultipleLines": [multiple_lines],

    "InternetService": [internet_service],

    "OnlineSecurity": [online_security],

    "OnlineBackup": [online_backup],

    "DeviceProtection": [device_protection],

    "TechSupport": [tech_support],

    "StreamingTV": [streaming_tv],

    "StreamingMovies": [streaming_movies],

    "Contract": [contract],

    "PaperlessBilling": [paperless_billing],

    "PaymentMethod": [payment_method],

    "MonthlyCharges": [monthly_charges],

    "TotalCharges": [total_charges]

})


# =========================================================
# CUSTOMER PREVIEW
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📋 Customer Preview
    </div>
    """,
    unsafe_allow_html=True
)

st.dataframe(
    customer_data,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

predict_button = st.button(
    "🔮  Predict Customer Churn",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # =====================================================
    # MAKE PREDICTION
    # =====================================================

    prediction = model.predict(
        customer_data
    )[0]


    # =====================================================
    # GET CHURN PROBABILITY
    # =====================================================

    probability = model.predict_proba(
        customer_data
    )[0][1]


    # =====================================================
    # RESULT SECTION
    # =====================================================

    st.divider()

    st.markdown(
        """
        <div class="section-title">
            📈 Prediction Result
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # RESULT MESSAGE
    # =====================================================

    if prediction == 1:

        st.error(
            "⚠️  CUSTOMER LIKELY TO CHURN"
        )

        st.warning(
            "The machine learning model predicts "
            "that this customer has a higher likelihood "
            "of leaving the service."
        )

    else:

        st.success(
            "✅  CUSTOMER LIKELY TO STAY"
        )

        st.info(
            "The machine learning model predicts "
            "that this customer is more likely "
            "to continue using the service."
        )


    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Prediction",
            "CHURN" if prediction == 1 else "STAY"
        )


    with col2:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


    with col3:

        stay_probability = 1 - probability

        st.metric(
            "Stay Probability",
            f"{stay_probability * 100:.2f}%"
        )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.markdown(
        "### Churn Probability"
    )

    st.progress(
        float(probability)
    )

    st.caption(
        f"Model confidence for churn: "
        f"{probability * 100:.2f}%"
    )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    if probability >= 0.75:

        st.error(
            "🔴 Very high churn probability. "
            "The customer may require immediate attention."
        )

    elif probability >= 0.50:

        st.warning(
            "🟠 High churn probability. "
            "The customer may be at risk of leaving."
        )

    elif probability >= 0.25:

        st.info(
            "🟡 Moderate churn probability. "
            "The customer should be monitored."
        )

    else:

        st.success(
            "🟢 Low churn probability. "
            "The customer is less likely to leave."
        )



# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 13px;">
        <strong>Customer Churn Prediction System</strong>
        <br><br>
        Built with Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)