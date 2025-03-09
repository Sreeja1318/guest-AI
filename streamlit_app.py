import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from datetime import date, datetime
from pymongo import MongoClient
import os
from openpyxl import load_workbook

# Load necessary models
encoder = joblib.load('C:\\Users\\LENOVO\\Downloads\\infosys\\encoder.pkl')
label_encoder = joblib.load('C:\\Users\\LENOVO\\Downloads\\infosys\\label_encoder.pkl')
model = joblib.load('C:\\Users\\LENOVO\\Downloads\\infosys\\xgb_model_dining.pkl')
features = list(pd.read_excel('features.xlsx')[0])

# Database connection (MongoDB Atlas)
client = MongoClient("mongodb+srv://sreejachiluveru744:sreeja13@cluster0.urhcm.mongodb.net/hotel_guests?retryWrites=true&w=majority")
db = client["hotel_guests"]
bookings_collection = db["new_bookings"]

# Streamlit UI
st.set_page_config(page_title="🍽️ AI-Powered Dining Experience", layout="wide")
st.title("🌟 Welcome to Smart Dining! 🍛")
st.markdown("Your AI-powered personalized dining experience awaits!")

# User Input Section
st.sidebar.header("📋 Booking Details")
has_customer_id = st.sidebar.radio("Do you have a Customer ID?", ["Yes", "No"])
customer_id = st.sidebar.text_input("Enter Customer ID") if has_customer_id == "Yes" else random.randint(10001, 99999)
name = st.sidebar.text_input("Enter your name")
age = st.sidebar.number_input("Enter your age", min_value=18, max_value=100, step=1)
stayers = st.sidebar.slider("How many people are staying?", 1, 5, 1)
checkin_date = st.sidebar.date_input("Check-in Date", min_value=date.today())
checkout_date = st.sidebar.date_input("Check-out Date", min_value=checkin_date)
preferred_cuisine = st.sidebar.selectbox("Choose Preferred Cuisine", ["South Indian", "North Indian", "Multi"])
booking_points = st.sidebar.radio("Use Booking Points?", ["Yes", "No"])

# Submit Button
if st.sidebar.button("🔍 Get Dish Recommendations"):
    if name:
        new_data = pd.DataFrame([{ 
            'customer_id': customer_id,
            'Preferred Cusine': preferred_cuisine,
            'age': age,
            'check_in_date': checkin_date,
            'check_out_date': checkout_date,
            'booked_through_points': 1 if booking_points == "Yes" else 0,
            'number_of_stayers': stayers,
            'check_in_day': checkin_date.weekday(),
            'check_out_day': checkout_date.weekday(),
            'check_in_month': checkin_date.month,
            'check_out_month': checkout_date.month,
            'stay_duration': (checkout_date - checkin_date).days
        }])

        # Load and merge feature data
        customer_features = pd.read_excel('customer_features.xlsx')
        cuisine_features = pd.read_excel('cuisine_features.xlsx')
        customer_dish = pd.read_excel('customer_dish.xlsx')
        cuisine_dish = pd.read_excel('cuisine_dish.xlsx')
        cuisine_diversity = pd.read_excel('cuisine_diversity.xlsx')
        customer_behavior_features = pd.read_excel('customer_behavior_features.xlsx')

        new_data["customer_id"] = new_data["customer_id"].astype(int)
        customer_features["customer_id"] = customer_features["customer_id"].astype(int)
        customer_dish["customer_id"] = customer_dish["customer_id"].astype(int)

        new_data = new_data.merge(customer_features, on='customer_id', how='left')
        new_data = new_data.merge(cuisine_features, on='Preferred Cusine', how='left')
        new_data = new_data.merge(customer_dish, on='customer_id', how='left')
        new_data = new_data.merge(cuisine_dish, on='Preferred Cusine', how='left')
        new_data = new_data.merge(cuisine_diversity, on='customer_id', how='left')
        new_data = new_data.merge(customer_behavior_features, on='customer_id', how='left')
        new_data.drop(columns=['customer_id', 'check_in_date', 'check_out_date'], inplace=True)

        # Handle categorical encoding
        original_categorical_cols = list(encoder.feature_names_in_)
        for col in original_categorical_cols:
            if col not in new_data.columns:
                new_data[col] = "unknown"

        extra_cols = [col for col in new_data.columns if col not in original_categorical_cols and new_data[col].dtype == 'object']
        new_data.drop(columns=extra_cols, inplace=True)

        encoded_test = encoder.transform(new_data[original_categorical_cols])
        encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(original_categorical_cols))
        new_data = pd.concat([new_data.drop(columns=original_categorical_cols), encoded_test_df], axis=1)

        for feature in features:
            if feature not in new_data.columns:
                new_data[feature] = 0

        new_data = new_data[features]

        # Predict
        y_pred_prob = model.predict_proba(new_data)
        dish_names = label_encoder.classes_
        top_3_indices = np.argsort(-y_pred_prob, axis=1)[:, :3]
        top_3_dishes = dish_names[top_3_indices]

        st.success(f"✅ Booking Confirmed for {name}!")
        st.subheader("🔹 Top Recommended Dishes for You:")
        st.markdown(f"1️⃣ **{top_3_dishes[0, 0]}** 🍽️")
        st.markdown(f"2️⃣ **{top_3_dishes[0, 1]}** 🍛")
        st.markdown(f"3️⃣ **{top_3_dishes[0, 2]}** 🍲")

        booking_data = {
            "Customer ID": customer_id,
            "Name": name,
            "Age": age,
            "Number of Stayers": stayers,
            "Check-in Date": checkin_date,
            "Check-out Date": checkout_date,
            "Preferred Cuisine": preferred_cuisine,
            "Booked Through Points": booking_points,
            "Recommended Dish 1": top_3_dishes[0, 0],
            "Recommended Dish 2": top_3_dishes[0, 1],
            "Recommended Dish 3": top_3_dishes[0, 2],
            "Timestamp": datetime.now()
        }

        try:
            file_path = "dining_info.xlsx"
            if os.path.exists(file_path):
                existing_df = pd.read_excel(file_path)
                updated_df = pd.concat([existing_df, pd.DataFrame([booking_data])], ignore_index=True)
            else:
                updated_df = pd.DataFrame([booking_data])
            updated_df.to_excel(file_path, index=False)
        except Exception as e:
            st.error(f"Error saving booking to Excel: {e}")

        st.markdown("\n\n🎉 **Exclusive Discounts Just for You!** 🎉")
        thali_dishes = [dish for dish in top_3_dishes[0] if "thali" in dish.lower()]
        other_dishes = [dish for dish in top_3_dishes[0] if "thali" not in dish.lower()]

        if thali_dishes:
            st.markdown(f"🌟 **20% OFF** on {', '.join(thali_dishes)}! 🎊")
        if other_dishes:
            st.markdown(f"✨ **15% OFF** on {', '.join(other_dishes)}! 🍴")
        st.markdown("📩 Check your email for your exclusive discount coupon!")
    else:
        st.warning("⚠️ Please enter your name to proceed!")

# Debugging: Show all bookings
if st.button("Show All Bookings"):
    try:
        df = pd.read_excel('dining_info.xlsx')
        st.subheader("📋 All Saved Bookings")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load bookings: {e}")
