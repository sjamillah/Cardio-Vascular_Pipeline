import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from src.model import load_latest_model
from src.preprocessing import preprocess_single_datapoint

def predict_risk(input_data):
    """
    Make a prediction for a single data point
    
    Args:
        input_data (dict): Dictionary containing the input data features
        
    Returns:
        dict: Prediction results including risk level and probabilities
    """
    try:
        # Preprocess the input data
        processed_data = preprocess_single_datapoint(input_data)
        
        # Load the latest model
        model = load_latest_model()
        
        # Make prediction
        prediction_probs = model.predict(processed_data)
        prediction = np.argmax(prediction_probs, axis=1)[0]
        
        # Map numeric prediction to risk level description
        risk_levels = {
            0: "Low Risk",
            1: "Medium Risk",
            2: "High Risk"
        }
        
        # Format probabilities
        probabilities = {f"Class {i}": float(prob) for i, prob in enumerate(prediction_probs[0])}
        
        result = {
            "risk_level": int(prediction),
            "risk_description": risk_levels[prediction],
            "probabilities": probabilities
        }
        
        return result
    except Exception as e:
        return {"error": str(e)}

def generate_visualization(feature_name, data_path):
    """
    Generate visualization for a specific feature
    
    Args:
        feature_name (str): Name of the feature to visualize
        data_path (str): Path to the CSV data file
        
    Returns:
        str: Base64 encoded image
    """
    try:
        # Load data
        data = pd.read_csv(data_path, delimiter=";")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Different visualizations based on feature type
        if feature_name in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
            # For numerical features - histogram with risk level breakdown
            risk_levels = np.select(
                [
                    (data['ap_hi'] <= 140) & (data['cholesterol'] <= 1) & (data['gluc'] <= 1),
                    ((data['ap_hi'] > 140) & (data['ap_hi'] <= 160)) | (data['cholesterol'] == 2) | (data['gluc'] == 2),
                    (data['ap_hi'] > 160) | (data['cholesterol'] > 2) | (data['gluc'] > 2)
                ],
                [0, 1, 2],
                default=1
            )
            
            # Create DataFrame with feature and risk level
            df = pd.DataFrame({
                'feature': data[feature_name],
                'risk_level': risk_levels
            })
            
            # Plot histograms by risk level
            for risk in [0, 1, 2]:
                subset = df[df['risk_level'] == risk]['feature']
                plt.hist(subset, alpha=0.5, bins=30, label=f'Risk Level {risk}')
            
            plt.title(f'Distribution of {feature_name} by Risk Level')
            plt.xlabel(feature_name)
            plt.ylabel('Count')
            plt.legend()
            
        elif feature_name in ['gender', 'cholesterol', 'gluc']:
            # For categorical features - bar chart
            risk_levels = np.select(
                [
                    (data['ap_hi'] <= 140) & (data['cholesterol'] <= 1) & (data['gluc'] <= 1),
                    ((data['ap_hi'] > 140) & (data['ap_hi'] <= 160)) | (data['cholesterol'] == 2) | (data['gluc'] == 2),
                    (data['ap_hi'] > 160) | (data['cholesterol'] > 2) | (data['gluc'] > 2)
                ],
                [0, 1, 2],
                default=1
            )
            
            # Create cross-tabulation
            cross_tab = pd.crosstab(data[feature_name], risk_levels)
            
            # Plot stacked bar chart
            cross_tab.plot(kind='bar', stacked=True)
            plt.title(f'{feature_name} vs Risk Level')
            plt.xlabel(feature_name)
            plt.ylabel('Count')
            plt.legend(title='Risk Level')
            
        else:
            plt.text(0.5, 0.5, f"Visualization for {feature_name} not implemented",
                     horizontalalignment='center', verticalalignment='center')
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    except Exception as e:
        plt.close()
        return str(e)

def generate_feature_importance(data_path):
    """
    Generate feature importance visualization
    
    Args:
        data_path (str): Path to the CSV data file
        
    Returns:
        str: Base64 encoded image
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from src.preprocessing import load_and_preprocess_data
        
        # Load and preprocess data
        X, Y = load_and_preprocess_data(data_path, drop_first=False)
        
        # Train a simple random forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, Y)
        
        # Get feature importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    except Exception as e:
        plt.close()
        return str(e)
