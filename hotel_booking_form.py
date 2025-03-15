import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from datetime import date, datetime
from pymongo import MongoClient
import os
from openpyxl import load_workbook
import smtplib
from email.message import EmailMessage

# Load necessary models
encoder = joblib.load('C:\\Users\\LENOVO\\Downloads\\infosys\\encoder.pkl')
label_encoder = joblib.load('C:\\Users\\LENOVO\\Downloads\\infosys\\label_encoder.pkl')
model = joblib.load('C:\\Users\\LENOVO\\Downloads\\infosys\\xgb_model_dining.pkl')
features = list(pd.read_excel('features.xlsx')[0])

# Database connection (MongoDB Atlas)
client = MongoClient("mongodb+srv://sreejachiluveru744:sreeja13@cluster0.urhcm.mongodb.net/hotel_guests?retryWrites=true&w=majority")
db = client["hotel_guests"]
bookings_collection = db["new_bookings"]

# Email configuration
EMAIL_ADDRESS = 'sreejachiluveru744@gmail.com'  # Replace with your email
EMAIL_PASSWORD = 'sjikkresbbdesvuw'   # Replace with your email password

# Function to generate a unique coupon code
def generate_coupon_code(customer_id):
    return f"DISCOUNT-{customer_id}-{random.randint(1000, 9999)}"

def send_email(to_email, subject, body):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("❌ Failed to authenticate. Please check your email credentials.")
    except smtplib.SMTPException as e:
        st.error(f"❌ Failed to send email: {e}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
    return False

# Streamlit UI
st.set_page_config(page_title="🍽️ AI-Powered Dining Experience", layout="wide")
st.title("🌟 Welcome to Smart Dining! 🍛")
st.markdown("Your AI-powered personalized dining experience awaits!")

# Important Note for Email Functionality
with st.expander("⚠️ Important Note for Email Functionality"):
    st.markdown("""
    To ensure the email functionality works correctly, please follow these steps:
    
    1. **If using Gmail:**
       - Go to your [Google Account Security Settings](https://myaccount.google.com/security).
       - Enable **"Less secure app access"** (if using a personal Gmail account).
       - Alternatively, if you have **2-Factor Authentication (2FA)** enabled, generate an **App Password** and use it instead of your regular password.

    2. **Replace Email Credentials:**
       - In the code, replace `EMAIL_ADDRESS` and `EMAIL_PASSWORD` with your actual email credentials.
    
    3. **Test the Email Functionality:**
       - Ensure the email sending feature is working by making a test booking.
    """)

# User Input Section
st.sidebar.header("📋 Booking Details")
has_customer_id = st.sidebar.radio("Do you have a Customer ID?", ["Yes", "No"], key="customer_id_radio")

if has_customer_id == "No":
    customer_id = random.randint(10001, 99999)  # Generate new ID
    st.sidebar.text(f"Generated Customer ID: {customer_id}")
else:
    customer_id_input = st.sidebar.text_input("Enter Customer ID", key="customer_id_input")
    if customer_id_input.isdigit():
        customer_id = int(customer_id_input)
    else:
        st.error("Please enter a valid numeric Customer ID.")
        customer_id = None

name = st.sidebar.text_input("Enter your name")
age = st.sidebar.number_input("Enter your age", min_value=18, max_value=100, step=1)
stayers = st.sidebar.slider("How many people are staying?", 1, 5, 1)
checkin_date = st.sidebar.date_input("Check-in Date", min_value=date.today())
checkout_date = st.sidebar.date_input("Check-out Date", min_value=checkin_date)
preferred_cuisine = st.sidebar.selectbox("Choose Preferred Cuisine", ["South Indian", "North Indian", "Multi"])
booking_points = st.sidebar.radio("Use Booking Points?", ["Yes", "No"])
email = st.sidebar.text_input("Enter your email address to receive booking confirmation")

# Submit Button
if st.sidebar.button("🔍 Get Dish Recommendations"):
    if name and email:
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
        top_3_dishes = [dish_names[i] for i in top_3_indices[0]]  # Ensure unique dishes

        st.success(f"✅ Booking Confirmed for {name}!")
        st.subheader("🔹 Top Recommended Dishes for You:")
        st.markdown(f"1️⃣ **{top_3_dishes[0]}** 🍽️")
        st.markdown(f"2️⃣ **{top_3_dishes[1]}** 🍛")
        st.markdown(f"3️⃣ **{top_3_dishes[2]}** 🍲")

        # Generate a unique coupon code
        coupon_code = generate_coupon_code(customer_id)

        booking_data = {
            "Customer ID": customer_id,
            "Name": name,
            "Age": age,
            "Number of Stayers": stayers,
            "Check-in Date": checkin_date,
            "Check-out Date": checkout_date,
            "Preferred Cuisine": preferred_cuisine,
            "Booked Through Points": booking_points,
            "Recommended Dish 1": top_3_dishes[0],
            "Recommended Dish 2": top_3_dishes[1],
            "Recommended Dish 3": top_3_dishes[2],
            "Coupon Code": coupon_code,
            "Timestamp": datetime.now()
        }

        try:
            file_path = "bookings_data.xlsx"
            if os.path.exists(file_path):
                existing_df = pd.read_excel(file_path)
                updated_df = pd.concat([existing_df, pd.DataFrame([booking_data])], ignore_index=True)
            else:
                updated_df = pd.DataFrame([booking_data])
            updated_df.to_excel(file_path, index=False)
        except Exception as e:
            st.error(f"Error saving booking to Excel: {e}")

        # Send email confirmation with coupon code
        email_subject = "Your Booking Confirmation"
        email_body = f"""
        Dear {name},

        Thank you for booking with us! Here are your booking details:

        Customer ID: {customer_id}
        Name: {name}
        Age: {age}
        Number of Stayers: {stayers}
        Check-in Date: {checkin_date}
        Check-out Date: {checkout_date}
        Preferred Cuisine: {preferred_cuisine}
        Booked Through Points: {booking_points}

        Top Recommended Dishes:
        1. {top_3_dishes[0]}
        2. {top_3_dishes[1]}
        3. {top_3_dishes[2]}

        Exclusive Discounts:
        - 20% OFF on {top_3_dishes[0]}!
        - 15% OFF on {top_3_dishes[1]} and {top_3_dishes[2]}!

        Your Exclusive Coupon Code: **{coupon_code}**

        We look forward to serving you!

        Best regards,
        Smart Dining Team
        """
        if send_email(email, email_subject, email_body):
            st.success("📩 Booking confirmation email sent successfully!")
        else:
            st.error("Failed to send email. Please check your email credentials and settings.")

        st.markdown("\n\n🎉 **Exclusive Discounts Just for You!** 🎉")
        thali_dishes = [dish for dish in top_3_dishes if "thali" in dish.lower()]
        other_dishes = [dish for dish in top_3_dishes if "thali" not in dish.lower()]

        if thali_dishes:
            st.markdown(f"🌟 **20% OFF** on {', '.join(thali_dishes)}! 🎊")
        if other_dishes:
            st.markdown(f"✨ **15% OFF** on {', '.join(other_dishes)}! 🍴")
        st.markdown(f"📩 Check your email for your exclusive discount coupon! Your Coupon Code: **{coupon_code}**")
    else:
        st.warning("⚠️ Please enter your name and email to proceed!")

# Debugging: Show all bookings
if st.button("Show All Bookings"):
    try:
        df = pd.read_excel('bookings_data.xlsx')
        st.subheader("📋 All Saved Bookings")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load bookings: {e}")