import streamlit as st
import pandas as pd
import joblib
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="wide"
)

# -----------------------------
# Load Model & Encoders
# -----------------------------
model = joblib.load("model.pkl2")
encoders = joblib.load("label_encoders.pkl2")

# -----------------------------
# Distance Function
# -----------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)

    a = (
        sin(dlat/2)**2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon/2)**2
    )

    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background:#F5F7FB;
}

/* Hide Streamlit Menu */
#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Title */
h1{
text-align:center;
color:#1565C0;
font-size:42px;
}

/* Section Heading */
.section{
background:#1565C0;
color:white;
padding:12px;
border-radius:10px;
font-size:22px;
font-weight:bold;
margin-top:25px;
margin-bottom:18px;
}

/* Button */
.stButton>button{
width:100%;
height:55px;
border-radius:12px;
background: #2563EB;
color:white;
font-size:24px;
font-weight:bold;
border:none;
}

.stButton>button:hover{
background:#0D47A1;
}

/* Success Box */
.success{
background:#E3F2FD;
padding:18px;
border-radius:12px;
font-size:24px;
font-weight:bold;
text-align:center;
color:#1B5E20;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🍲 Food Delivery Time Prediction")

st.markdown(
    """
    <div style="
        text-align:center;
        font-size:22px;
        font-weight:600;
        color:#374151;
        margin-bottom:25px;
    ">
        This AI-powered application predicts the estimated food delivery time
using Machine Learning.Fill in the delivery details below and click <b style="color:#2563EB;">Predict Delivery Time</b>.
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 👤 Delivery Person Details
# ==========================================

st.markdown(
    '<div class="section">👤 Delivery Person Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    

    age = st.number_input(
        "**Delivery Person Age**",
        min_value=18,
        max_value=60,
        value=None,
        placeholder="Enter Age"
    )

with col2:

    rating = st.number_input(
        "**Delivery Person Rating**",
        min_value=1.0,
        max_value=5.0,
        step=0.1,
        value=None,
        placeholder="Enter Rating"
    )

# ==========================================
# 📍 Restaurant & Delivery Location
# ==========================================

st.markdown(
    '<div class="section">📍 Restaurant & Delivery Location</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    restaurant_lat = st.number_input(
        "**Restaurant Latitude**",
        value=None,
        step=0.000001,
        format="%.6f",
        placeholder="Enter Restaurant Latitude"
    )

    restaurant_long = st.number_input(
        "**Restaurant Longitude**",
        value=None,
        step=0.000001,
        format="%.6f",
        placeholder="Enter Restaurant Longitude"
    )

with col2:

    delivery_lat = st.number_input(
        "**Delivery Latitude**",
        value=None,
        step=0.000001,
        format="%.6f",
        placeholder="Enter Delivery Latitude"
    )

    delivery_long = st.number_input(
        "**Delivery Longitude**",
        value=None,
        step=0.000001,
        format="%.6f",
        placeholder="Enter Delivery Longitude"
    )
    # ==========================================
# 📅 Order Details
# ==========================================

st.markdown(
    '<div class="section">📅 Order Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    order_day = st.number_input(
        "**Order Day**",
        min_value=1,
        max_value=31,
        value=None,
        placeholder="Enter Day"
    )

    order_month = st.number_input(
        "**Order Month**",
        min_value=1,
        max_value=12,
        value=None,
        placeholder="Enter Month"
    )


with col2:

    order_hour = st.number_input(
        "**Order Hour**",
        min_value=0,
        max_value=23,
        value=None,
        placeholder="0-23"
    )

    order_minute = st.number_input(
        "**Order Minute**",
        min_value=0,
        max_value=59,
        value=None,
        placeholder="0-59"
    )

    pickup_hour = st.number_input(
        "**Pickup Hour**",
        min_value=0,
        max_value=23,
        value=None,
        placeholder="0-23"
    )

    pickup_minute = st.number_input(
        "**Pickup Minute**",
        min_value=0,
        max_value=59,
        value=None,
        placeholder="0-59"
    )


# ==========================================
# 🌦 Weather & Vehicle Details
# ==========================================

st.markdown(
    '<div class="section">🌦 Weather & Vehicle Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    weather = st.selectbox(
        "**Weather Conditions**",
        encoders["Weatherconditions"].classes_,
        index=None,
        placeholder="Select Weather"
    )

    traffic = st.selectbox(
        "**Road Traffic Density**",
        encoders["Road_traffic_density"].classes_,
        index=None,
        placeholder="Select Traffic"
    )

    vehicle_condition = st.number_input(
        "**Vehicle Condition**",
        min_value=0,
        max_value=3,
        value=None,
        placeholder="0-3"
    )

    order_type = st.selectbox(
        "**Type of Order**",
        encoders["Type_of_order"].classes_,
        index=None,
        placeholder="Select Order Type"
    )

with col2:

    vehicle_type = st.selectbox(
        "**Type of Vehicle**",
        encoders["Type_of_vehicle"].classes_,
        index=None,
        placeholder="Select Vehicle"
    )

    multiple_deliveries = st.number_input(
        "**Multiple Deliveries**",
        min_value=0,
        max_value=3,
        value=None,
        placeholder="0-3"
    )

    festival = st.selectbox(
        "**Festival**",
        encoders["Festival"].classes_,
        index=None,
        placeholder="Select Festival"
    )

    city = st.selectbox(
        "**City**",
        encoders["City"].classes_,
        index=None,
        placeholder="Select City"
    )

st.markdown("---")

predict = st.button("🚚 Predict Delivery Time", use_container_width=True)

# ==========================================
# ==========================================
# 🚀 Prediction
# ==========================================

if predict:

    # Check all fields
    if (
        age is None or
        rating is None or
        restaurant_lat is None or
        restaurant_long is None or
        delivery_lat is None or
        delivery_long is None or
        order_day is None or
        order_month is None or
        order_hour is None or
        order_minute is None or
        pickup_hour is None or
        pickup_minute is None or
        weather is None or
        traffic is None or
        vehicle_condition is None or
        order_type is None or
        vehicle_type is None or
        multiple_deliveries is None or
        festival is None or
        city is None
    ):

        st.warning("⚠️ Please fill all fields.")

    else:

        # Calculate weekday automatically
        from datetime import datetime

        order_weekday = datetime(
            2026,
            int(order_month),
            int(order_day)
        ).weekday()

        # Encode categorical features
        weather = encoders["Weatherconditions"].transform([weather])[0]
        traffic = encoders["Road_traffic_density"].transform([traffic])[0]
        order_type = encoders["Type_of_order"].transform([order_type])[0]
        vehicle_type = encoders["Type_of_vehicle"].transform([vehicle_type])[0]
        festival = encoders["Festival"].transform([festival])[0]
        city = encoders["City"].transform([city])[0]

        # Create dataframe
        input_df = pd.DataFrame([[
            age,
            rating,
            restaurant_lat,
            restaurant_long,
            delivery_lat,
            delivery_long,
            weather,
            traffic,
            vehicle_condition,
            order_type,
            vehicle_type,
            multiple_deliveries,
            festival,
            city,
            order_day,
            order_month,
            order_weekday,
            order_hour,
            order_minute,
            pickup_hour,
            pickup_minute
        ]], columns=[
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
            "Weatherconditions",
            "Road_traffic_density",
            "Vehicle_condition",
            "Type_of_order",
            "Type_of_vehicle",
            "multiple_deliveries",
            "Festival",
            "City",
            "Order_Day",
            "Order_Month",
            "Order_Weekday",
            "Order_Hour",
            "Order_Minute",
            "Pickup_Hour",
            "Pickup_Minute"
        ])

        # Predict
        prediction = model.predict(input_df)[0]

        # Distance
        distance = calculate_distance(
            restaurant_lat,
            restaurant_long,
            delivery_lat,
            delivery_long
        )

        # -----------------------------
                # -----------------------------
        # Result
        # -----------------------------

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div style="
                    background:#16A34A;
                    color:white;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    box-shadow:0px 3px 10px rgba(0,0,0,0.2);
                ">
                    <h4>🍔 Estimated Delivery Time</h4>
                    <h2>{prediction:.2f} min</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div style="
                    background:#16A34A;
                    color:white;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    box-shadow:0px 3px 10px rgba(0,0,0,0.2);
                ">
                    <h4>📍 Delivery Distance</h4>
                    <h2>{distance:.2f} km</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("")

        if prediction <= 20:
            msg = "🟢 Fast Delivery"
            bg = "#D1FAE5"
            color = "#065F46"

        elif prediction <= 40:
            msg = "🟡 Moderate Delivery"
            bg = "#FEF3C7"
            color = "#92400E"

        else:
            msg = "🔴 Slow Delivery"
            bg = "#FEE2E2"
            color = "#991B1B"

        st.markdown(
            f"""
            <div style="
                background:{bg};
                color:{color};
                text-align:center;
                padding:15px;
                border-radius:10px;
                font-size:22px;
                font-weight:bold;
                margin-top:10px;
            ">
                {msg}
            </div>
            """,
            unsafe_allow_html=True
        )
        # -----------------------------
    
                # ==========================
        # Download Report
        # ==========================

        report = pd.DataFrame({
            "Predicted Time (min)": [round(prediction, 2)],
            "Distance (km)": [round(distance, 2)]
        })

        csv = report.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="Delivery_Prediction_Report.csv",
            mime="text/csv"
        )
        
        st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:10px;">
    <p style="color:gray; font-size:15px;">
        🍔 <b>Food Delivery Time Prediction</b><br>
        Developed using <b>Python, Streamlit & XGBoost</b><br>
    </p>
</div>
""", unsafe_allow_html=True)