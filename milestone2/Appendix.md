Appendix for Milestone 2: Hotel Booking System with Personalized Dining Recommendations

Introduction This appendix provides an in-depth explanation of the implementation carried out in Milestone 2, where we developed a hotel booking system that personalizes dining recommendations based on customer preferences. This system was built upon the model developed in Milestone 1, utilizing machine learning techniques and database integration to enhance user experience.

System Architecture
The system comprises the following key components:

User Input Form: Captures customer details including name, email, check-in/check-out dates, age, number of stayers, preferred cuisine, and booking preferences.
Data Processing Module: Transforms raw input into the format required by the trained model.
Machine Learning Model: Predicts the top three recommended dishes for the user.
Database Integration: Stores booking details in MongoDB.
Email Notification System: Sends confirmation emails with booking details and discount coupons.
Functional Implementation

User Input Collection
Customers either input an existing Customer ID or receive a new generated one.
Name, email, check-in/check-out dates, and other details are collected.
Data Transformation
Converts dates to weekday and month features.
Merges multiple feature datasets related to customer age, behavior, loyalty, and cuisine preferences.
Encodes categorical data using a pre-trained OneHotEncoder.
Ensures model compatibility by adding missing feature columns and reordering as per the training dataset.
Model Prediction
Loads the trained XGBoost model and predicts dish recommendations.
Uses a label encoder to map predictions to dish names.
Extracts the top three recommended dishes.
Database Storage
Stores new booking details in MongoDB under the new_bookings!
 collection.
Email Confirmation & Coupon Generation
Generates a unique coupon code for the user.
Sends an email containing booking confirmation and dining discount coupon using SMTP.
Booking Confirmation Display
Displays confirmation message along with recommended dishes and special requests (if any).
Results and Observations

The system successfully captures user input and stores it in MongoDB.
The XGBoost model provides personalized dish recommendations based on the user’s profile.
Email notifications are sent with confirmation details and discount coupons.
The web interface is intuitive and user-friendly.
Conclusion Milestone 2 successfully built upon the model developed in Milestone 1 by integrating it into a functional hotel booking system. The system enhances user experience through personalized dining recommendations and seamless booking confirmation.

Output Images



![Image_Alt](https://github.com/Sreeja1318/guest-AI/blob/f9bdc59c5b1dd38a0cebd2a848e132d313bd552a/milestone2/hotelbooking.png)



![Image_Alt](https://github.com/Sreeja1318/guest-AI/blob/fa876d6dbc8a9883ca80a30a43675fd82bdcf8f1/milestone2/bookinconfimation.png)

