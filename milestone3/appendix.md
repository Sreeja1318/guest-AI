Appendix: Milestone 3 - Hotel Review Analysis System

Overview
Milestone 3 involved integrating Together AI embeddings, Pinecone for vector storage, and Streamlit for building an interactive UI. The project focused on processing hotel customer reviews, generating embeddings, storing them in a Pinecone index, retrieving relevant reviews, analyzing sentiments, and providing a manager-friendly review analysis dashboard.

Customer Review Submission UI
Features:
Customers submit hotel reviews in real-time.
Sentiment Analysis:
Used TextBlob to calculate sentiment score.
If a negative review is submitted and the customer is currently staying, an automated email with room number, review, and sentiment score is sent to the hotel manager.
Manager Review Analysis UI
Built Using: Streamlit
Key Functionalities:
Query Input: Allows managers to ask review-related questions.
Relevant Review Retrieval: Uses Pinecone similarity search.
Summarization: Together AI generates a professional summary of customer sentiments.
Word Cloud Generation: Highlights frequently mentioned words based on the manager's query.
Data Download: Managers can download the sentiment analysis report as a .txt file.
Final Outcomes
✅ Efficient storage and retrieval of customer reviews using Pinecone. ✅ AI-powered summarization for quick managerial insights. ✅ Real-time sentiment detection and manager notifications. ✅ User-friendly Streamlit dashboard for customer and manager interactions. ✅ Visualization enhancements using word clouds and sentiment scores.

UI Outcomes Images

1.CUSTOMER REVIEWS
![Image_Alt](https://github.com/Sreeja1318/guest-AI/blob/64f889681c551ecd75ce4e2431b29d680273406a/milestone3/review.png)


2.MANAGER REVIEW ANALYSIS
![Image_Alt](https://github.com/Sreeja1318/guest-AI/blob/262b77d6ec0e3a9489984ac022b1cbf6bfea6ee2/milestone3/managerreviews.png)
