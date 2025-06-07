#!/usr/bin/env python3
"""
Irrigation Prediction Model

This model predicts whether a plant needs irrigation based on several environmental factors:
- Temperature
- Humidity 
- Soil Moisture
- Soil Type
- Crop Type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def main():
    """Main function to run the irrigation prediction model"""
    
    # Load data (replace file path with your correct path)
    try:
        df = pd.read_csv('your_data_file.csv')  # Replace with your data file path
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("Error: Data file not found. Please update the file path.")
        print("Creating sample data for demonstration...")
        df = create_sample_data()
    
    # Display first 5 rows of data
    print("\nFirst 5 rows of data:")
    print(df.head())
    
    # General information about the data
    print(f"\nGeneral information about the data:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Data Preparation
    data_irrigation = df.copy()
    
    # Create new column "Irrigation Needed" based on Moisture value
    # We assume that Moisture < 400 means the soil is dry and needs irrigation
    data_irrigation['Irrigation Needed'] = data_irrigation['Moisture'].apply(lambda x: 1 if x < 400 else 0)
    
    print(f"\nDistribution of irrigation needs:")
    print(data_irrigation['Irrigation Needed'].value_counts())
    print(f"Percentage of plants needing irrigation: {data_irrigation['Irrigation Needed'].mean():.2%}")
    
    # Encode categorical columns
    label_encoders_irrigation = {}
    categorical_columns_irrigation = ['Soil Type', 'Crop Type']
    
    for col in categorical_columns_irrigation:
        le = LabelEncoder()
        data_irrigation[col] = le.fit_transform(data_irrigation[col])
        label_encoders_irrigation[col] = le
        
        print(f"\nEncoding {col}:")
        for i, class_name in enumerate(le.classes_):
            print(f"{class_name} -> {i}")
    
    # Data Exploration
    explore_data(data_irrigation)
    
    # Model Training
    features_irrigation = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type']
    X_irr = data_irrigation[features_irrigation]
    y_irr = data_irrigation['Irrigation Needed']
    
    print(f"\nData shape:")
    print(f"Independent variables (X): {X_irr.shape}")
    print(f"Dependent variable (y): {y_irr.shape}")
    
    # Split the data
    X_irr_train, X_irr_test, y_irr_train, y_irr_test = train_test_split(
        X_irr, y_irr, test_size=0.2, random_state=42, stratify=y_irr
    )
    
    print(f"\nData split:")
    print(f"Training data: {X_irr_train.shape[0]} samples")
    print(f"Testing data: {X_irr_test.shape[0]} samples")
    print(f"Irrigation need ratio in training: {y_irr_train.mean():.2%}")
    print(f"Irrigation need ratio in testing: {y_irr_test.mean():.2%}")
    
    # Train the classification model
    model_irrigation = RandomForestClassifier(n_estimators=100, random_state=42)
    model_irrigation.fit(X_irr_train, y_irr_train)
    print("\nModel trained successfully!")
    
    # Model Evaluation
    y_irr_pred = model_irrigation.predict(X_irr_test)
    y_irr_pred_proba = model_irrigation.predict_proba(X_irr_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_irr_test, y_irr_pred)
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_irr_test, y_irr_pred, 
                              target_names=['No Irrigation', 'Needs Irrigation']))
    
    # Plot results
    plot_results(y_irr_test, y_irr_pred, y_irr_pred_proba, model_irrigation, features_irrigation)
    
    # Save results
    save_results(y_irr_pred, y_irr_pred_proba, y_irr_test)
    
    # Example prediction
    demonstrate_prediction(model_irrigation, label_encoders_irrigation, features_irrigation)
    
    # Results summary
    print_summary(data_irrigation, X_irr_train, X_irr_test, accuracy, model_irrigation, features_irrigation)

def create_sample_data():
    """Create sample data for demonstration if no data file is found"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Temparature': np.random.normal(25, 8, n_samples),
        'Humidity': np.random.normal(60, 15, n_samples),
        'Moisture': np.random.normal(450, 100, n_samples),
        'Soil Type': np.random.choice(['Clay', 'Sandy', 'Loamy', 'Silty'], n_samples),
        'Crop Type': np.random.choice(['Wheat', 'Corn', 'Rice', 'Tomato', 'Potato'], n_samples)
    }
    
    # Ensure values are within realistic ranges
    data['Temparature'] = np.clip(data['Temparature'], 10, 45)
    data['Humidity'] = np.clip(data['Humidity'], 20, 95)
    data['Moisture'] = np.clip(data['Moisture'], 200, 800)
    
    return pd.DataFrame(data)

def explore_data(data_irrigation):
    """Explore and visualize the data"""
    # Plot distribution of numerical variables
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    numeric_columns = ['Temparature', 'Humidity', 'Moisture']
    for i, col in enumerate(numeric_columns):
        axes[i].hist(data_irrigation[col], bins=20, alpha=0.7, color='skyblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Plot irrigation needs
    axes[3].bar(['No Irrigation', 'Needs Irrigation'], 
               data_irrigation['Irrigation Needed'].value_counts().values,
               color=['lightgreen', 'lightcoral'])
    axes[3].set_title('Distribution of Irrigation Needs')
    axes[3].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = data_irrigation[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Irrigation Needed']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix Between Variables')
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results(y_test, y_pred, y_pred_proba, model, features):
    """Plot model results and visualizations"""
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Irrigation', 'Needs Irrigation'],
                yticklabels=['No Irrigation', 'Needs Irrigation'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance in the Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Prediction probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[:, 1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold (0.5)')
    plt.title('Distribution of Prediction Probabilities for Irrigation Need')
    plt.xlabel('Probability of Needing Irrigation')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(y_pred, y_pred_proba, y_test):
    """Save prediction results to CSV"""
    # Generate ID for each prediction
    irrigation_ids = [str(uuid.uuid4()) for _ in range(len(y_pred))]
    
    # Save results to CSV
    irrigation_result_df = pd.DataFrame({
        'ID': irrigation_ids,
        'Predicted_Irrigation_Needed': y_pred,
        'Prediction_Probability': y_pred_proba[:, 1],  # Probability of needing irrigation
        'Actual_Irrigation_Needed': y_test.values
    })
    
    irrigation_csv_path = "irrigation_predictions_with_id.csv"
    irrigation_result_df.to_csv(irrigation_csv_path, index=False)
    
    print(f"\nResults saved to: {irrigation_csv_path}")
    print(f"First 10 predictions:")
    print(irrigation_result_df.head(10))

def demonstrate_prediction(model, label_encoders, features):
    """Demonstrate how to use the model for new predictions"""
    def predict_irrigation_need(temperature, humidity, moisture, soil_type, crop_type):
        """
        Predict irrigation need for new data
        
        Parameters:
        temperature: Temperature
        humidity: Humidity
        moisture: Soil moisture
        soil_type: Soil type (text)
        crop_type: Crop type (text)
        
        Returns:
        prediction: 0 (no irrigation needed) or 1 (irrigation needed)
        probability: Probability of needing irrigation
        """
        
        # Encode text data
        try:
            soil_encoded = label_encoders['Soil Type'].transform([soil_type])[0]
            crop_encoded = label_encoders['Crop Type'].transform([crop_type])[0]
        except ValueError as e:
            return f"Error: Soil type or crop type not found in training data. {e}"
        
        # Create input data for prediction
        input_data = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return prediction, probability
    
    # Get unique values for soil types and crops
    available_soil_types = label_encoders['Soil Type'].classes_
    available_crop_types = label_encoders['Crop Type'].classes_
    
    print(f"\nExample prediction:")
    print(f"Available soil types: {list(available_soil_types)}")
    print(f"Available crop types: {list(available_crop_types)}")
    
    # Example prediction
    example_temp = 30
    example_humidity = 65
    example_moisture = 350  # Low value indicating need for irrigation
    example_soil = available_soil_types[0]
    example_crop = available_crop_types[0]
    
    pred, prob = predict_irrigation_need(example_temp, example_humidity, example_moisture, 
                                       example_soil, example_crop)
    
    print(f"\nInput data:")
    print(f"Temperature: {example_temp}°C")
    print(f"Humidity: {example_humidity}%")
    print(f"Soil moisture: {example_moisture}")
    print(f"Soil type: {example_soil}")
    print(f"Crop type: {example_crop}")
    
    print(f"\nResult:")
    print(f"Prediction: {'Needs irrigation' if pred == 1 else 'No irrigation needed'}")
    print(f"Probability of needing irrigation: {prob:.2%}")

def print_summary(data_irrigation, X_train, X_test, accuracy, model, features):
    """Print model summary"""
    print("="*60)
    print("IRRIGATION PREDICTION MODEL SUMMARY")
    print("="*60)
    
    print(f"\n📊 Data Information:")
    print(f"   • Total samples: {len(data_irrigation)}")
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Testing samples: {len(X_test)}")
    print(f"   • Percentage of plants needing irrigation: {data_irrigation['Irrigation Needed'].mean():.2%}")
    
    print(f"\n🎯 Model Performance:")
    print(f"   • Accuracy: {accuracy:.2%}")
    print(f"   • Model type: Random Forest")
    print(f"   • Number of trees: 100")
    
    print(f"\n🔧 Features Used:")
    for feature in features:
        importance = model.feature_importances_[features.index(feature)]
        print(f"   • {feature}: {importance:.4f}")
    
    print(f"\n📋 Irrigation Criteria Used:")
    print(f"   • Soil moisture < 400 = Needs irrigation")
    print(f"   • Soil moisture ≥ 400 = No irrigation needed")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()