import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QSpinBox, QSlider,
                           QPushButton, QProgressBar, QGroupBox, QFormLayout,
                           QDoubleSpinBox, QTableWidget, QTableWidgetItem, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor, QFont
import joblib
import json

class AirlineSatisfactionPredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Airline Passenger Satisfaction Predictor")
        self.setMinimumSize(1200, 900)  # Increased window size
        
        # Load models
        self.models = {}
        model_names = ['Random Forest', 'KNN', 'SVC', 'XGBoost']
        for name in model_names:
            try:
                model_path = f'{name.lower().replace(" ", "_")}_model.joblib'
                self.models[name] = joblib.load(model_path)
            except:
                print(f"Could not load {name} model")
        
        # Load model metrics
        try:
            with open('model_metrics.json', 'r') as f:
                self.model_metrics = json.load(f)
        except:
            self.model_metrics = {}
        
        # Create a scroll area
        scroll = QScrollArea()
        self.setCentralWidget(scroll)
        
        # Create main widget and layout
        main_widget = QWidget()
        scroll.setWidget(main_widget)
        scroll.setWidgetResizable(True)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow, QScrollArea, QWidget {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 16px;  /* Base font size */
                padding: 8px;
            }
            QGroupBox {
                color: #ffffff;
                font-size: 20px;  /* Larger font for group titles */
                font-weight: bold;
                border: 2px solid #3d3d3d;
                border-radius: 10px;
                margin-top: 20px;
                padding: 15px;
            }
            QGroupBox::title {
                padding: 0 15px;
            }
            QComboBox, QSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 5px;
                padding: 10px;
                min-height: 40px;  /* Increased height */
                min-width: 200px;  /* Minimum width */
                font-size: 16px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                width: 15px;
                height: 15px;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 15px 30px;  /* Increased padding */
                border-radius: 8px;
                font-weight: bold;
                font-size: 18px;  /* Larger font */
                min-height: 50px;  /* Increased height */
                min-width: 200px;  /* Minimum width */
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                text-align: center;
                color: white;
                font-size: 16px;
                min-height: 40px;  /* Increased height */
                max-height: 40px;
                min-width: 400px;  /* Minimum width */
            }
            QProgressBar::chunk {
                background-color: #1565c0;
                border-radius: 6px;
            }
            QTableWidget {
                background-color: #2b2b2b;
                color: white;
                gridline-color: #3d3d3d;
                font-size: 16px;
                border: 1px solid #505050;
                border-radius: 5px;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: white;
                padding: 10px;
                border: 1px solid #505050;
                font-size: 16px;
                font-weight: bold;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 25px;
                height: 25px;
            }
        """)

        # Create main layout
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)  # Increased spacing
        main_layout.setContentsMargins(30, 30, 30, 30)  # Increased margins

        # Create form layouts for different feature groups
        self.create_passenger_info_group(main_layout)
        self.create_travel_info_group(main_layout)
        self.create_service_ratings_group(main_layout)
        self.create_model_comparison_group(main_layout)

    def create_passenger_info_group(self, parent_layout):
        group = QGroupBox("Passenger Information")
        layout = QFormLayout()
        layout.setSpacing(15)  # Increased spacing between form elements
        layout.setContentsMargins(15, 25, 15, 15)  # Increased margins

        # Gender selection
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Male", "Female"])
        layout.addRow("Gender:", self.gender_combo)

        # Customer Type selection
        self.customer_type_combo = QComboBox()
        self.customer_type_combo.addItems(["Loyal Customer", "disloyal Customer"])
        layout.addRow("Customer Type:", self.customer_type_combo)

        # Age input
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 120)
        self.age_spin.setValue(30)
        layout.addRow("Age:", self.age_spin)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_travel_info_group(self, parent_layout):
        group = QGroupBox("Travel Information")
        layout = QFormLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 25, 15, 15)

        # Travel Type selection
        self.travel_type_combo = QComboBox()
        self.travel_type_combo.addItems(["Business travel", "Personal Travel"])
        layout.addRow("Type of Travel:", self.travel_type_combo)

        # Class selection
        self.class_combo = QComboBox()
        self.class_combo.addItems(["Business", "Eco", "Eco Plus"])
        layout.addRow("Class:", self.class_combo)

        # Flight Distance
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(0, 5000)
        self.distance_spin.setValue(500)
        layout.addRow("Flight Distance:", self.distance_spin)

        # Delay inputs
        self.departure_delay_spin = QSpinBox()
        self.departure_delay_spin.setRange(0, 1440)
        layout.addRow("Departure Delay (minutes):", self.departure_delay_spin)

        self.arrival_delay_spin = QSpinBox()
        self.arrival_delay_spin.setRange(0, 1440)
        layout.addRow("Arrival Delay (minutes):", self.arrival_delay_spin)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_service_ratings_group(self, parent_layout):
        group = QGroupBox("Service Ratings (1-5)")
        layout = QFormLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 25, 15, 15)

        self.service_ratings = {}
        services = [
            "Inflight wifi service",
            "Departure/Arrival time convenient",
            "Ease of Online booking",
            "Gate location",
            "Food and drink",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Inflight service",
            "Cleanliness"
        ]

        for service in services:
            spin = QSpinBox()
            spin.setRange(1, 5)
            spin.setValue(3)
            layout.addRow(f"{service}:", spin)
            self.service_ratings[service] = spin

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_model_comparison_group(self, parent_layout):
        group = QGroupBox("Model Comparison")
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(15, 25, 15, 15)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setFont(QFont('Arial', 16))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.models.keys()))
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # Predict button
        button_layout = QHBoxLayout()
        self.predict_button = QPushButton("Predict Satisfaction")
        button_layout.addWidget(self.predict_button, alignment=Qt.AlignCenter)
        self.predict_button.clicked.connect(self.predict_satisfaction)
        layout.addLayout(button_layout)

        # Add prediction confidence bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMinimum(0)
        self.confidence_bar.setMaximum(100)
        layout.addWidget(self.confidence_bar, alignment=Qt.AlignCenter)

        # Add prediction label
        self.prediction_label = QLabel("Prediction: ")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setFont(QFont('Arial', 16))
        layout.addWidget(self.prediction_label)

        # Create metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setRowCount(len(self.models))
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        # Set table dimensions
        self.metrics_table.setMinimumHeight(200)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        
        # Populate metrics table
        for i, (model_name, metrics) in enumerate(self.model_metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(model_name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{metrics['accuracy']:.4f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{metrics['precision']:.4f}"))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{metrics['recall']:.4f}"))
            self.metrics_table.setItem(i, 4, QTableWidgetItem(f"{metrics['f1']:.4f}"))
        
        self.metrics_table.resizeColumnsToContents()
        layout.addWidget(self.metrics_table)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def predict_satisfaction(self):
        selected_model = self.model_combo.currentText()
        model = self.models.get(selected_model)
        
        if model is None:
            self.prediction_label.setText(f"Error: {selected_model} model not loaded")
            return

        # Gather all input values with exact column names matching training data
        input_data = {
            'Gender': [1 if self.gender_combo.currentText() == 'Male' else 0],
            'Customer Type': [1 if self.customer_type_combo.currentText() == 'Loyal Customer' else 0],
            'Age': [self.age_spin.value()],
            'Type of Travel': [1 if self.travel_type_combo.currentText() == 'Business travel' else 0],
            'Class': [2 if self.class_combo.currentText() == 'Business' else 1 if self.class_combo.currentText() == 'Eco Plus' else 0],
            'Flight Distance': [self.distance_spin.value()],
            'Inflight wifi service': [self.service_ratings["Inflight wifi service"].value()],
            'Departure/Arrival time convenient': [self.service_ratings["Departure/Arrival time convenient"].value()],
            'Ease of Online booking': [self.service_ratings["Ease of Online booking"].value()],
            'Gate location': [self.service_ratings["Gate location"].value()],
            'Food and drink': [self.service_ratings["Food and drink"].value()],
            'Online boarding': [self.service_ratings["Online boarding"].value()],
            'Seat comfort': [self.service_ratings["Seat comfort"].value()],
            'Inflight entertainment': [self.service_ratings["Inflight entertainment"].value()],
            'On-board service': [self.service_ratings["On-board service"].value()],
            'Leg room service': [self.service_ratings["Leg room service"].value()],
            'Baggage handling': [self.service_ratings["Baggage handling"].value()],
            'Checkin service': [self.service_ratings["Checkin service"].value()],
            'Inflight service': [self.service_ratings["Inflight service"].value()],
            'Cleanliness': [self.service_ratings["Cleanliness"].value()],
            'Departure Delay in Minutes': [self.departure_delay_spin.value()],
            'Arrival Delay in Minutes': [self.arrival_delay_spin.value()]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Make prediction
        try:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0]
            
            # Update UI
            result = "satisfied" if prediction[0] == 1 else "neutral or dissatisfied"
            confidence = max(probability) * 100
            
            self.confidence_bar.setValue(int(confidence))
            self.prediction_label.setText(f"Prediction ({selected_model}): {result} (Confidence: {confidence:.1f}%)")
            
        except Exception as e:
            self.prediction_label.setText(f"Error making prediction: {str(e)}")
            print(f"Prediction error: {str(e)}")  # Print error for debugging

def main():
    app = QApplication(sys.argv)
    window = AirlineSatisfactionPredictor()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 