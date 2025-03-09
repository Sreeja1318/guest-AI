
from pymongo import MongoClient
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

client = MongoClient("mongodb+srv://sreejachiluveru744:sreeja13@cluster0.urhcm.mongodb.net/hotel_guests?retryWrites=true&w=majority")

db = client["hotel_guests"]

collection = db["dining_info"]

df_from_mongo = pd.DataFrame(list(collection.find()))

df = df_from_mongo.copy()

df['check_in_date'] = pd.to_datetime(df['check_in_date'])
df['check_out_date'] = pd.to_datetime(df['check_out_date'])
df['order_time'] = pd.to_datetime(df['order_time'])

df['check_in_day'] = df['check_in_date'].dt.dayofweek  # Monday=0, Sunday=6
df['check_out_day'] = df['check_out_date'].dt.dayofweek
df['check_in_month'] = df['check_in_date'].dt.month
df['check_out_month'] = df['check_out_date'].dt.month
df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days

df[['booked_through_points','price_for_1']].groupby('booked_through_points').mean()

features_df = df[df['order_time']<'2024-01-01']

# I can direct features
train_df = df[(df['order_time']>='2024-01-01')&(df['order_time']<='2024-10-01')]

test_df = df[(df['order_time']>'2024-10-01')] # - pseudo prediction dataset


#1-------
customer_features = features_df.groupby("customer_id").agg(
    total_orders_per_customer=("transaction_id", "count"),
    #avg_spend_per_customer=("price_for_1", "mean"),
    fav_dish_per_customer=("dish", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
    most_preferred_cuisine=("Preferred Cusine", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
).reset_index()

customer_features.to_excel('customer_features.xlsx',index=False)

#2-----------
age_features =features_df.groupby('age').agg(
    age_pref_cuisine=("Preferred Cusine", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
    #fav_dish_as_per_age=("dish", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),

).reset_index()
age_features.to_excel('age_features.xlsx', index=False)




# 🌟3-- Cuisine-Level Aggregations
# Calculate cuisine-level aggregations
cuisine_features = features_df.groupby("Preferred Cusine").agg(
    #avg_price_per_cuisine=("price_for_1", "mean"),
    total_orders_per_cuisine=("transaction_id", "count"),
     avg_spend_per_cuisine=('price_for_1', 'mean') 
).reset_index()

cuisine_features.to_excel('cuisine_features.xlsx',index=False)

#4------
cuisine_diversity = features_df.groupby('customer_id')['Preferred Cusine'].nunique().reset_index()
avg_price_per_cuisine=("price_for_1", "mean")
cuisine_diversity.rename(columns={'Preferred Cusine': 'cuisine_diversity_score'}, inplace=True)
cuisine_diversity.to_excel('cuisine_diversity.xlsx',index=False)





 #5------#customer_behavior _features
customer_behavior_features = features_df.groupby("customer_id").agg(
    #unique_dishes_ordered=("dish", "nunique"),
    unique_cuisines_ordered=("Preferred Cusine", "nunique"),
    most_common_dish=("dish", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
    most_common_cuisine=("Preferred Cusine", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
).reset_index()
customer_behavior_features.to_excel('customer_behavior_features.xlsx',index=False)






train_df = train_df.merge(customer_features, on='customer_id', how='left')
train_df = train_df.merge(cuisine_features, on='Preferred Cusine', how='left')
train_df = train_df.merge(customer_behavior_features, on="customer_id", how="left")
train_df = train_df.merge(cuisine_diversity, on="customer_id", how="left")
train_df = train_df.merge(age_features, on="age", how="left")

print("Columns after merging:", train_df.columns)



train_df.drop(['_id','transaction_id','customer_id','price_for_1',
               'Qty','order_time','check_in_date','check_out_date'],axis=1,inplace=True)

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Select categorical columns for one-hot encoding
categorical_cols =['Preferred Cusine','fav_dish_per_customer','most_common_dish','most_common_cuisine','age_pref_cuisine','most_preferred_cuisine']

print("Columns before encoding:", train_df.columns)

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


# Apply transformation
encoded_array = encoder.fit_transform(train_df[categorical_cols])
print("Available columns in train_df:", train_df.columns)

print(encoded_array)

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

import joblib

# Store the encoder
joblib.dump(encoder, 'encoder.pkl')

# Load the encoder when needed
loaded_encoder = joblib.load('encoder.pkl')


# Concatenate with the original DataFrame
train_df = pd.concat([train_df.drop(columns=categorical_cols), encoded_df], axis=1)

train_df.columns

test_df = test_df.merge(customer_features, on='customer_id', how='left')
test_df = test_df.merge(cuisine_features, on='Preferred Cusine', how='left')
test_df =test_df.merge(age_features, on="age", how="left")
test_df = test_df.merge(customer_behavior_features, on="customer_id", how="left")
test_df = test_df.merge(cuisine_diversity, on="customer_id", how="left")

test_df.drop(['_id','transaction_id','customer_id','price_for_1',
               'Qty','order_time','check_in_date','check_out_date'],axis=1,inplace=True)


encoded_test = encoder.transform(test_df[categorical_cols])

# Convert to DataFrame
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate with test_df
test_df = pd.concat([test_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

train_df

train_df = train_df.dropna(subset=['dish'])

# Encode the target column 'dish' using LabelEncoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_df['dish'] = label_encoder.fit_transform(train_df['dish'])

joblib.dump(label_encoder, 'label_encoder.pkl')

# Split into features (X) and target (y)
X_train = train_df.drop(columns=['dish'])  # Features
y_train = train_df['dish']

test_df = test_df.dropna(subset=['dish'])

# Encode 'dish' using the SAME LabelEncoder from training
test_df['dish'] = label_encoder.transform(test_df['dish']) 

from sklearn.metrics import accuracy_score, log_loss

X_test = test_df.drop(columns=['dish'])  # Features
y_test = test_df['dish']

xgb_model = XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    learning_rate=0.01, #0.2 0.1(0.13850931)  0.01,100(0.14596273) # Increase LR to speed up learning
    n_estimators=100,  # 110(0.1360242) 115(0.13472) 113(0.137881) 114,0.1(0.13850931)  100,0.1(0.139130434) More trees can improve accuracy
    max_depth=6,  #6 Reduce depth to prevent overfitting
    random_state=9,#9
    subsample=0.4,#0.3  # Slightly higher subsample
    colsample_bytree=0.3#0.5
    )

# Train the model
xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, 'xgb_model_dining.pkl')
pd.DataFrame(X_train.columns).to_excel('features.xlsx')