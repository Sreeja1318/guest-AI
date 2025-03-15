import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pinecone import Pinecone
from langchain_together import TogetherEmbeddings
from together import Together
import io


# ✅ Set Streamlit Page Config at the Start
st.set_page_config(layout="wide")

# ✅ Set API Keys
if not os.getenv("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = "45cc35a9dc8621b295401d3b841e56b82afef4d426a90e3c2f72fde63927ab92"

# ✅ Load Data
@st.cache_data
def load_data():
    return pd.read_excel("reviews_data.xlsx")

df = load_data()


# ✅ Function to Generate Query-Based Word Cloud from Together API Response
def generate_wordcloud_from_api(filtered_reviews, query):
    if filtered_reviews.empty:
        st.sidebar.warning("⚠️ No relevant reviews found for this query!")
        return

    concatenated_reviews = " ".join(filtered_reviews["Review"].tolist())
    client = Together()
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{
            "role": "user",
            "content": f"Extract the most frequent and relevant words from these reviews based on the query: '{query}'. Here are the reviews: {concatenated_reviews}. Return only a comma-separated list of words."
        }]
    )
    

# ✅ Streamlit Layout
st.title("🛎️ Hotel Manager's Review Analysis Tool")
st.markdown("💡 **Analyze customer feedback, track sentiments, and generate key insights!**")

# ✅ Manager's Query Input
st.subheader("🎙️ Ask Anything About Customer Reviews")
query = st.text_input("e.g., 'What do customers say about our breakfast service?'")

#PINECONE IITIALIZATION----------
pc = Pinecone(api_key="pcsk_3GJ2wg_R6FhBarLn8wJNR1rqTvUv8FNkwkXSa4V2cujUrVGW3uadaw6YXLtQAWVhBMTMp9")
index = pc.Index(host="https://hotel-reviews-izeoe32.svc.aped-4627-b74a.pinecone.io")

if st.button("🔎 Get Insights"):
    if query:
        embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
        query_embedding = embeddings.embed_query(query)

        results = index.query(
            vector=query_embedding,
            top_k=10,  
            include_metadata=True
        )

        matches = results["matches"]
        matched_ids = [int(match["metadata"]["review_id"]) for match in matches]

        if "review_id" in df.columns and "Review" in df.columns:
            req_df = df[df["review_id"].isin(matched_ids)]

            if not req_df.empty:
                concatenated_reviews = " ".join(req_df["Review"].tolist())

                # ✅ Generate Answer using Together AI
                client = Together()
                response = client.chat.completions.create(
                    model="meta-llama/Llama-Vision-Free",
                    messages=[{
                        "role": "user",
                        "content": f"Based on these customer reviews, answer this manager's query: {query}. Here are the relevant reviews: {concatenated_reviews}. Provide a concise and professional summary that is elaborated."
                    }]
                )
                
                answer = response.choices[0].message.content
                st.subheader("💡 Insightful Summary")
                st.write(answer)
                
                output = io.BytesIO()
                output.write(answer.encode())
                st.download_button("📥 Download Report", data=output, file_name="review_analysis.txt", mime="text/plain")
                
                # ✅ Generate Query-Based Word Cloud in Sidebar
                generate_wordcloud_from_api(req_df, query)
            else:
                st.warning("⚠️ No matching reviews found for this query.")
        else:
            st.warning("⚠️ 'review_id' or 'Review' column missing in dataset.")
    else:
        st.warning("⚠️ Please enter a query.")
