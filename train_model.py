import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

def train_and_save_models():
    print("Loading data...")
    df = pd.read_csv("train.csv")
    
    # Drop unnecessary columns
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    
    # Handle missing values in 'Arrival Delay in Minutes'
    df['Arrival Delay in Minutes'].fillna(df['Departure Delay in Minutes'], inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
        
    # Split features and target
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create smaller subset for SVC
    X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, train_size=5000, random_state=42)
    
    # Initialize models
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'X_train': X_train,
            'y_train': y_train
        },
        'KNN': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'X_train': X_train,
            'y_train': y_train
        },
        'SVC': {
            'model': SVC(probability=True, random_state=42),
            'X_train': X_train_small,  # Use smaller dataset for SVC
            'y_train': y_train_small
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42),
            'X_train': X_train,
            'y_train': y_train
        }
    }
    
    # Dictionary to store model metrics
    model_metrics = {}
    
    print("\nTraining and evaluating models...")
    for name, model_dict in models.items():
        print(f"\nTraining {name}...")
        if name == 'SVC':
            print("(Using smaller dataset for faster training)")
        
        model = model_dict['model']
        X_train_current = model_dict['X_train']
        y_train_current = model_dict['y_train']
        
        model.fit(X_train_current, y_train_current)
        
        # Make predictions on the full test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        model_metrics[name] = metrics
        
        # Save model
        joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.joblib')
        print(f"{name} metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
    
    # Save metrics to JSON file
    with open('model_metrics.json', 'w') as f:
        json.dump(model_metrics, f, indent=4)
    
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_models() 