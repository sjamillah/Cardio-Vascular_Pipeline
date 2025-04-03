import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Define constants for categorical features
CULTURAL_BELIEF_CATEGORIES = ['Never', 'Occasionally', 'Frequently']
TREATMENT_ADHERENCE_CATEGORIES = ['Low', 'Medium', 'High']
DISTANCE_CATEGORIES = ['Near', 'Moderate', 'Far']
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

def load_and_preprocess_data(data_source, is_file=True, drop_first=True):
    """
    Load and preprocess the cardiovascular dataset
    
    Args:
        data_source (str or pd.DataFrame): Path to CSV or DataFrame
        is_file (bool): Whether the data_source is a file path
        drop_first (bool): Whether to drop first category in dummy variables
    
    Returns:
        tuple: Processed features (X) and risk level (Y)
    """
    try:
        # Load data based on source type
        if is_file:
            data = pd.read_csv(data_source, delimiter=";")
        else:
            data = data_source.copy()

        relevant_features = ['age', 'height', 'weight', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        
        # If coming from file, add these features and create a proper copy
        if is_file:
            # Create a proper copy to avoid SettingWithCopyWarning
            data_filtered = data[relevant_features + ['cardio']].copy()
            
            # Set random seed for reproducibility
            np.random.seed(42)
            
            # Add categorical features correctly
            data_filtered['cultural_belief_score'] = np.random.choice(
                CULTURAL_BELIEF_CATEGORIES, 
                len(data_filtered)
            )
            
            data_filtered['treatment_adherence'] = np.random.choice(
                TREATMENT_ADHERENCE_CATEGORIES, 
                len(data_filtered)
            )
            
            data_filtered['distance_to_healthcare'] = np.random.choice(
                DISTANCE_CATEGORIES, 
                len(data_filtered)
            )

            # Calculate risk level
            data_filtered['risk_level'] = np.select(
                [
                    (data_filtered['ap_hi'] <= 140) & (data_filtered['cholesterol'] <= 1) & (data_filtered['gluc'] <= 1),
                    ((data_filtered['ap_hi'] > 140) & (data_filtered['ap_hi'] <= 160)) | (data_filtered['cholesterol'] == 2) | (data_filtered['gluc'] == 2),
                    (data_filtered['ap_hi'] > 160) | (data_filtered['cholesterol'] > 2) | (data_filtered['gluc'] > 2)
                ],
                [0, 1, 2],
                default=1
            )

            # Drop the cardio column
            data_filtered.drop('cardio', axis=1, inplace=True)
        else:
            # For manually added data, assume risk_level is already present
            # Make a proper copy to avoid warnings
            data_filtered = data.copy()

        # Define complete set of relevant features
        relevant_features = [
            'age', 'height', 'weight', 'gender', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'cultural_belief_score',
            'treatment_adherence', 'distance_to_healthcare'
        ]
        
        # Ensure all required columns exist
        for feature in relevant_features:
            if feature not in data_filtered.columns:
                logger.warning(f"Missing feature: {feature}. Adding with default values.")
                if feature == 'cultural_belief_score':
                    data_filtered[feature] = CULTURAL_BELIEF_CATEGORIES[0]
                elif feature == 'treatment_adherence':
                    data_filtered[feature] = TREATMENT_ADHERENCE_CATEGORIES[0]
                elif feature == 'distance_to_healthcare':
                    data_filtered[feature] = DISTANCE_CATEGORIES[0]
                else:
                    data_filtered[feature] = 0
        
        # Select only the relevant features plus risk_level
        data_filtered = data_filtered[relevant_features + ['risk_level']]

        # One-hot encode categorical variables
        # For retraining, set drop_first=False to preserve all categories
        data_processed = pd.get_dummies(data_filtered, drop_first=drop_first)

        # If we're keeping all categories, we don't need the expected columns check
        if drop_first:
            # Check for expected columns after one-hot encoding
            expected_columns = [
                'age', 'height', 'weight', 'gender', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 
                'cultural_belief_score_Occasionally', 'cultural_belief_score_Frequently',
                'treatment_adherence_Medium', 'treatment_adherence_High',
                'distance_to_healthcare_Moderate', 'distance_to_healthcare_Far',
                'risk_level'
            ]
            
            # Ensure all expected columns exist after encoding
            for col in expected_columns:
                if col != 'risk_level' and col not in data_processed.columns:
                    logger.warning(f"Missing encoded column: {col}. Adding with zeros.")
                    data_processed[col] = 0

        X = data_processed.drop('risk_level', axis=1)
        Y = data_processed['risk_level']
        
        return X, Y
        
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {e}")
        raise

def preprocess_single_datapoint(data_dict):
    """
    Preprocess a single datapoint for prediction
    
    Args:
        data_dict (dict): Dictionary with input data features
    
    Returns:
        np.array: Preprocessed data ready for prediction
    """
    try:
        # Create a one-row dataframe from the input dictionary
        df = pd.DataFrame([data_dict])
        
        # CRITICAL FIX: We need to properly handle 'cholesterol' and 'gluc' features
        # In model training, these are one-hot encoded, but in prediction they remain numeric
        
        # Process cholesterol - convert to one-hot encoding
        if 'cholesterol' in df.columns:
            cholesterol_value = df['cholesterol'].iloc[0]
            # Create dummy variables (similar to pd.get_dummies with drop_first=False)
            df['cholesterol_1'] = (df['cholesterol'] == 1).astype(int)
            df['cholesterol_2'] = (df['cholesterol'] == 2).astype(int)
            df['cholesterol_3'] = (df['cholesterol'] == 3).astype(int)
            # Drop original column
            df.drop('cholesterol', axis=1, inplace=True)
        else:
            # Add default dummy columns if feature is missing
            df['cholesterol_1'] = 1  # Default to normal cholesterol
            df['cholesterol_2'] = 0
            df['cholesterol_3'] = 0
        
        # Process glucose - convert to one-hot encoding
        if 'gluc' in df.columns:
            gluc_value = df['gluc'].iloc[0]
            # Create dummy variables
            df['gluc_1'] = (df['gluc'] == 1).astype(int)
            df['gluc_2'] = (df['gluc'] == 2).astype(int)
            df['gluc_3'] = (df['gluc'] == 3).astype(int)
            # Drop original column
            df.drop('gluc', axis=1, inplace=True)
        else:
            # Add default dummy columns if feature is missing
            df['gluc_1'] = 1  # Default to normal glucose
            df['gluc_2'] = 0
            df['gluc_3'] = 0
        
        # Process cultural_belief_score
        if 'cultural_belief_score' in df.columns:
            df['cultural_belief_score_Never'] = (df['cultural_belief_score'] == 'Never').astype(int)
            df['cultural_belief_score_Occasionally'] = (df['cultural_belief_score'] == 'Occasionally').astype(int)
            df['cultural_belief_score_Frequently'] = (df['cultural_belief_score'] == 'Frequently').astype(int)
            df.drop('cultural_belief_score', axis=1, inplace=True)
        else:
            # Add default dummy columns if feature is missing
            df['cultural_belief_score_Never'] = 1  # Default to "Never"
            df['cultural_belief_score_Occasionally'] = 0
            df['cultural_belief_score_Frequently'] = 0
        
        # Process treatment_adherence
        if 'treatment_adherence' in df.columns:
            df['treatment_adherence_Low'] = (df['treatment_adherence'] == 'Low').astype(int)
            df['treatment_adherence_Medium'] = (df['treatment_adherence'] == 'Medium').astype(int)
            df['treatment_adherence_High'] = (df['treatment_adherence'] == 'High').astype(int)
            df.drop('treatment_adherence', axis=1, inplace=True)
        else:
            # Add default dummy columns if feature is missing
            df['treatment_adherence_Low'] = 1  # Default to "Low"
            df['treatment_adherence_Medium'] = 0
            df['treatment_adherence_High'] = 0
            
        # Process distance_to_healthcare
        if 'distance_to_healthcare' in df.columns:
            df['distance_to_healthcare_Near'] = (df['distance_to_healthcare'] == 'Near').astype(int)
            df['distance_to_healthcare_Moderate'] = (df['distance_to_healthcare'] == 'Moderate').astype(int)
            df['distance_to_healthcare_Far'] = (df['distance_to_healthcare'] == 'Far').astype(int)
            df.drop('distance_to_healthcare', axis=1, inplace=True)
        else:
            # Add default dummy columns if feature is missing
            df['distance_to_healthcare_Near'] = 1  # Default to "Near"
            df['distance_to_healthcare_Moderate'] = 0
            df['distance_to_healthcare_Far'] = 0
        
        # Define expected columns for model input - MUST match what the model was trained with
        expected_columns = [
            'age', 'height', 'weight', 'gender', 'ap_hi', 'ap_lo',
            'cholesterol_1', 'cholesterol_2', 'cholesterol_3',
            'gluc_1', 'gluc_2', 'gluc_3',
            'cultural_belief_score_Never', 'cultural_belief_score_Occasionally', 'cultural_belief_score_Frequently',
            'treatment_adherence_Low', 'treatment_adherence_Medium', 'treatment_adherence_High',
            'distance_to_healthcare_Near', 'distance_to_healthcare_Moderate', 'distance_to_healthcare_Far'
        ]
        
        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col} in input data. Adding with default value 0.")
                df[col] = 0
        
        # Reorder and select only expected columns
        df = df[expected_columns]
        
        # Print shape for debugging
        print(f"Preprocessed data shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Load the scaler that was used for the training data
        os.makedirs(MODELS_DIR, exist_ok=True)
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                # Apply scaling
                scaled_data = scaler.transform(df)
                print(f"Scaled data shape: {scaled_data.shape}")
                return scaled_data
            except Exception as e:
                logger.error(f"Error loading or applying scaler: {e}")
                logger.warning("Using unscaled data for prediction.")
                return df.values
        else:
            # If no scaler found, return the raw data (not ideal but allows for functionality)
            logger.warning("No scaler found at path: {}. Using unscaled data for prediction.".format(scaler_path))
            return df.values
    except Exception as e:
        logger.error(f"Error in preprocess_single_datapoint: {e}")
        raise

def scale_and_split_data(X, Y, test_size=0.3, val_size=0.2):
    """
    Scale features and split data into train, validation, and test sets.
    
    Args:
        X (pd.DataFrame): Features
        Y (pd.Series): Target variable
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of remaining data for validation set
    
    Returns:
        tuple: Scaled datasets and scaler object
    """
    try:
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=test_size, random_state=42)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=val_size/(1-test_size), random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler for future use
        os.makedirs(MODELS_DIR, exist_ok=True)
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, Y_train, Y_val, Y_test, scaler
    except Exception as e:
        logger.error(f"Error in scale_and_split_data: {e}")
        raise
