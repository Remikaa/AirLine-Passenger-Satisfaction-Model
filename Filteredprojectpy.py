import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import joblib
import os

s = StandardScaler()

def DataPreprocessing(features):
    # Compute Z-scores
    z_scores = zscore(features[['Age', 'Flight Distance', 'Arrival Delay in Minutes']])
    threshold = 3
    outliers = (np.abs(z_scores) > threshold)
    rows_to_keep = ~(outliers.any(axis=1))
    # Filter the DataFrame to remove outliers
    features = features[rows_to_keep].reset_index(drop=True)


    # Filling missing values with median as distribution of arrival delay was heavily skewed & its affected by outliers
    median_val = features['Arrival Delay in Minutes'].median()
    features['Arrival Delay in Minutes'] = features['Arrival Delay in Minutes'].fillna(median_val)

    #Replace categorical values with the mode
    columns_containing_0 = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Checkin service', 'Inflight service', 'Cleanliness']
    modes = features[columns_containing_0].mode()
    features[columns_containing_0] = features[columns_containing_0].replace({'0': modes})

    #Lot of features were skewed, therefore it will be appropriate to apply log transformation to them
    features['Flight Distance'] = np.log1p(features['Flight Distance'])


    # Label encoding the categorical features
    features['Gender'] = features['Gender'].map({'Male' : 0, 'Female' : 1})
    features['Customer Type'] = features['Customer Type'].map({'Loyal Customer' : 0, 'disloyal Customer' : 1})
    features['Type of Travel'] = features['Type of Travel'].map({'Personal Travel' : 0, 'Business travel' : 1})
    features['Class'] = features['Class'].map({'Business' : 2, 'Eco Plus' : 1, 'Eco' : 0})

    return features


# load dataset
train_df = pd.read_csv("train.csv")

#Dropping Departure Delay, Id and Unnamed column column
train_df.drop(['Unnamed: 0', 'id', 'Departure Delay in Minutes'], axis= 1, inplace= True)

# Data Preprocessing
train_df = DataPreprocessing(train_df)
train_df['satisfaction'] = train_df['satisfaction'].map({'neutral or dissatisfied' : 0, 'satisfied' : 1})

# split
X_train = train_df.drop('satisfaction', axis = 1)
y_train = train_df.satisfaction

X_train = s.fit_transform(X_train)

print("Data preprocessing ended successfully")

# importing Knn model & accuracy measures 
#from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors = 3)
#knn.fit(X_train, y_train)

#print("Knn model ended successfully")

# Import Logistic Regression function
#from sklearn.linear_model import LogisticRegression

# create and fit the model
#log_reg = LogisticRegression()
#log_reg.fit(X_train, y_train)

#print("Logisitic regression ended successfully")

# import the SVC library 
#from sklearn.svm import SVC
#Create and fit the model
#svm_model = SVC(kernel='linear')  
#svm_model.fit(X_train, y_train)
#print("SVC model ended successfully")

# import the RandomForest libary 
from sklearn.ensemble import RandomForestClassifier

# 1. Create and fit the model
if not os.path.exists("Random_forest_model.pkl"):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Random Forest model created successfully")
else:
    rf_model = joblib.load("Random_forest_model.pkl")
    print("Random Forest model loaded successfully")







import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler

# App title and description
st.title("✈️ Flight Satisfaction Predictor")
st.markdown("""
Predict passenger satisfaction based on flight experience metrics.  
Fill in the details below and click **Predict** to see the result.
""")

# Sidebar with info/instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Fill in passenger details
    2. Rate all flight experience aspects
    3. Click **Predict** button
    4. View results and recommendations
    """)
    
    st.header("About")
    st.markdown("""
    This app uses a machine learning model trained on airline passenger data  
    to predict satisfaction levels with **85% accuracy** (example metric).
    """)

# Main form for user input
with st.form("flight_details"):
    st.subheader("Passenger & Flight Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.slider("Age", 7, 85, 30, help="Passenger age between 7 and 85")
        travel_type = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
        flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
        flight_distance = st.slider("Flight Distance (miles)", 31, 5000, 500)
    
    with col2:
        arrival_delay = st.number_input("Arrival Delay (minutes)", 0, 1600, 0)
    
    st.subheader("Service Ratings (1-5 scale)")
    
    rating_cols = st.columns(4)
    
    with rating_cols[0]:
        wifi = st.slider("Wifi Quality", 1, 5, 3)
        time_convenient = st.slider("Timing Convenience", 1, 5, 3)
        online_booking = st.slider("Online Booking", 1, 5, 3)
        gate_location = st.slider("Gate Location", 1, 5, 3)
    
    with rating_cols[1]:
        food_drink = st.slider("Food & Drink", 1, 5, 3)
        online_boarding = st.slider("Online Boarding", 1, 5, 3)
        seat_comfort = st.slider("Seat Comfort", 1, 5, 3)
        entertainment = st.slider("Entertainment", 1, 5, 3)
    
    with rating_cols[2]:
        onboard_service = st.slider("On-board Service", 1, 5, 3)
        leg_room = st.slider("Leg Room", 1, 5, 3)
        baggage = st.slider("Baggage Handling", 1, 5, 4)
        checkin = st.slider("Check-in Service", 1, 5, 3)
    
    with rating_cols[3]:
        inflight_service = st.slider("Inflight Service", 1, 5, 4)
        cleanliness = st.slider("Cleanliness", 1, 5, 3)
    
    submitted = st.form_submit_button("Predict Satisfaction")

# When form is submitted
if submitted:
    # Prepare input data
    input_data = {
        'Gender': gender,
        'Customer Type': customer_type,
        'Age': age,
        'Type of Travel': travel_type,
        'Class': flight_class,
        'Flight Distance': flight_distance,
        'Inflight wifi service': wifi,
        'Departure/Arrival time convenient': time_convenient,
        'Ease of Online booking': online_booking,
        'Gate location': gate_location,
        'Food and drink': food_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': entertainment,
        'On-board service': onboard_service,
        'Leg room service': leg_room,
        'Baggage handling': baggage,
        'Checkin service': checkin,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Arrival Delay in Minutes': arrival_delay
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = DataPreprocessing(input_df)
    X = s.fit_transform(input_df)

    # Make prediction
    try:
        prediction = rf_model.predict(input_df)
        prediction_proba = rf_model.predict_proba(input_df)
        
        # Display results
        st.subheader("Prediction Result")
        #st.success(f"prediction ({prediction[0]})")
        if prediction[0] == 1:
            confidence = prediction_proba[0][1]
            st.success(f"✅ Predicted Satisfaction: **Satisfied** ({confidence*100:.1f}% confidence)")
            st.balloons()
        else:
            confidence = prediction_proba[0][0]
            st.error(f"❌ Predicted Satisfaction: **Neutral or Dissatisfied** ({confidence*100:.1f}% confidence)")
        
        # Feature importance visualization
        st.subheader("Key Influencing Factors")
        
        if hasattr(rf_model, 'feature_importances_'):
            features = input_df.columns
            importance = rf_model.feature_importances_
            
            # Create a DataFrame for visualization
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(10)
            
            st.bar_chart(importance_df.set_index('Feature'))
        else:
            st.info("Feature importance not available for this model type")
        
        # Recommendations for improvement
        if prediction[0] == 0:
            st.subheader("Recommendations to Improve Satisfaction")
            
            recs = []
            if arrival_delay > 30:
                recs.append("🕒 **Reduce arrival delays** (current: {} min)".format(arrival_delay))
            if inflight_service < 4:
                recs.append("✈️ **Improve inflight service** (current rating: {}/5)".format(inflight_service))
            if seat_comfort < 4:
                recs.append("💺 **Enhance seat comfort** (current rating: {}/5)".format(seat_comfort))
            if cleanliness < 4:
                recs.append("🧹 **Increase cleanliness** (current rating: {}/5)".format(cleanliness))
            if online_boarding < 4:
                recs.append("📱 **Streamline online boarding** (current rating: {}/5)".format(online_boarding))
            
            if recs:
                for rec in recs:
                    st.markdown("- " + rec)
            else:
                st.info("All key service aspects are rated highly. Consider operational factors.")
        
        # Download button for results
        csv = input_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction Details",
            data=csv,
            file_name="flight_satisfaction_prediction.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Add footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: #666;
    text-align: center;
    margin-top: 2rem;
}
</style>
<div class="footer">
    Flight Satisfaction Predictor | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)

# After training your model
if not os.path.exists("Random_forest_model.pkl"):
    joblib.dump(rf_model, 'Random_forest_model.pkl')