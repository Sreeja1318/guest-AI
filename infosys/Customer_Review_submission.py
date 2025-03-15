import streamlit as st
import pandas as pd
import datetime
import random
import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from together import Together
from langchain_together import TogetherEmbeddings
from textblob import TextBlob
import smtplib
from email.mime.text import MIMEText


# ✅ Initialize Pinecone
pc = Pinecone(api_key="pcsk_3GJ2wg_R6FhBarLn8wJNR1rqTvUv8FNkwkXSa4V2cujUrVGW3uadaw6YXLtQAWVhBMTMp9")
index = pc.Index(host="https://hotel-reviews-izeoe32.svc.aped-4627-b74a.pinecone.io")

# ✅ Set Together API Key
TOGETHER_API_KEY="45cc35a9dc8621b295401d3b841e56b82afef4d426a90e3c2f72fde63927ab92"
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
#embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-128M-8k-retrieval") #gpt----
#embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval") #gpt


# ✅ Define the Excel file
file_name = "C:\\Users\\LENOVO\\Downloads\\infosys\\reviews_data.xlsx"

# ✅ Load dataset or create if not exists
if os.path.exists(file_name):
    df = pd.read_excel(file_name)
else:
    df = pd.DataFrame(columns=["review_id", "customer_id", "review_date", "Review", "Rating", "review_date_numeric"])
    


# ✅ Function to generate a random 4-digit ID
def generate_id():
    return random.randint(1000, 9999)

# ✅ Function to send email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(room_number, review_text, sentiment_score):
    sender_email = "5@gmail.com"
    password = ""

    subject = "Negative Review Alert 🚨"
    body = f"Room Number: {room_number}\nReview: {review_text}\nSentiment Score: {sentiment_score}"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Connect to Gmail SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, message.as_string())


# ✅ Streamlit UI
st.set_page_config(page_title="Hotel Reviews", page_icon="⭐", layout="centered")
st.title("📢 Customer Review Submission")
st.markdown("Share your thoughts about our service! Your feedback helps us improve. 💬")

# User inputs
review_text = st.text_area("✍️ Write your review:", help="Describe your experience with us.")
rating = st.slider("⭐ Rate us (1-10)", 1, 10, 5)
room_number = st.text_input("🏨 Room Number (if staying):", "")

if st.button("✅ Submit Review", use_container_width=True):
    if review_text.strip():
        # ✅ Generate new review ID & customer ID
        new_review_id = generate_id()
        customer_id = generate_id()

        # ✅ Get current timestamp and convert it to numeric format
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        numeric_timestamp = int(datetime.datetime.now().timestamp())

        # ✅ Prepare new entry
        new_entry = pd.DataFrame([{
            "review_id": new_review_id,
            "customer_id": customer_id,
            "review_date": timestamp,
            "Review": review_text,
            "Rating": rating,
            "review_date_numeric": numeric_timestamp
        }])

        # ✅ Append to dataset & save
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_excel(file_name, index=False)

        # ✅ Generate Embeddings using Together AI
        review_embedding = embeddings.embed_query(review_text)

        # ✅ Store review in Pinecone
        index.upsert(
            vectors=[(str(new_review_id), review_embedding, {"review_id": new_review_id, "customer_id": customer_id, "rating": rating})]
        )

        # 🎉 Display Success Message with Review Details
        st.success("✅ Review submitted successfully!")

        st.markdown("### 📌 Submitted Review Details")
        st.write(f"**Review ID:** {new_review_id}")
        st.write(f"**Customer ID:** {customer_id}")
        st.write(f"**Date:** {timestamp}")
        st.write(f"**Review:** {review_text}")
        st.write(f"**Rating:** {rating} ⭐")

