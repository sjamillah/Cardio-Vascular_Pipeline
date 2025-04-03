import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import uvicorn
import os
import shutil
import joblib
from pathlib import Path
import traceback

# Configure logging with proper formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ensure project structure is properly imported
try:
    # First attempt - direct import
    from src.preprocessing import (
        load_and_preprocess_data, 
        preprocess_single_datapoint, 
        scale_and_split_data,

    )
    from src.model import fine_tune_model, load_latest_model, build_new_model
    from src.prediction import generate_visualization, generate_feature_importance
    from data.database import add_single_record, import_csv_to_db, get_training_data
    
    logger.info("Successfully imported modules using direct path")
except ImportError:
    # If that fails, try adjusting the path
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    
    # Now try imports again
    try:
        from src.preprocessing import (
            load_and_preprocess_data, 
            preprocess_single_datapoint, 
            scale_and_split_data
        )
        from src.model import fine_tune_model, load_latest_model
        from src.prediction import generate_visualization, generate_feature_importance
        from data.database import add_single_record, import_csv_to_db, get_training_data
        
        logger.info("Successfully imported modules using adjusted path")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        raise

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Define available categorical options
CULTURAL_BELIEF_OPTIONS = ['Never', 'Occasionally', 'Frequently']
TREATMENT_ADHERENCE_OPTIONS = ['Low', 'Medium', 'High']
DISTANCE_HEALTHCARE_OPTIONS = ['Near', 'Moderate', 'Far']

# Cardiovascular Risk Prediction Input Model
class CardiovascularRiskInput(BaseModel):
    age: int = Field(..., ge=0, description="Age in days")
    height: float = Field(..., gt=0, description="Height in cm")
    weight: float = Field(..., gt=0, description="Weight in kg")
    gender: int = Field(..., ge=0, le=1, description="Gender (0: Female, 1: Male)")
    ap_hi: int = Field(..., description="Systolic blood pressure")
    ap_lo: int = Field(..., description="Diastolic blood pressure")
    cholesterol: int = Field(..., ge=1, le=3, description="Cholesterol level (1: normal, 2: above normal, 3: well above normal)")
    gluc: int = Field(..., ge=1, le=3, description="Glucose level (1: normal, 2: above normal, 3: well above normal)")
    cultural_belief_score: str = Field(..., description="Cultural belief impact (Categorical)")
    treatment_adherence: str = Field(..., description="Treatment adherence level")
    distance_to_healthcare: str = Field(..., description="Distance to healthcare")
    
    # Add validators to ensure categorical variables have valid options
    @validator('cultural_belief_score')
    def validate_cultural_belief(cls, v):
        if v not in CULTURAL_BELIEF_OPTIONS:
            raise ValueError(f"cultural_belief_score must be one of {CULTURAL_BELIEF_OPTIONS}")
        return v
    
    @validator('treatment_adherence')
    def validate_treatment_adherence(cls, v):
        if v not in TREATMENT_ADHERENCE_OPTIONS:
            raise ValueError(f"treatment_adherence must be one of {TREATMENT_ADHERENCE_OPTIONS}")
        return v
    
    @validator('distance_to_healthcare')
    def validate_distance_to_healthcare(cls, v):
        if v not in DISTANCE_HEALTHCARE_OPTIONS:
            raise ValueError(f"distance_to_healthcare must be one of {DISTANCE_HEALTHCARE_OPTIONS}")
        return v
    
    @validator('ap_hi', 'ap_lo')
    def validate_blood_pressure(cls, v):
        if v <= 0:
            raise ValueError(f"Blood pressure values must be positive")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "age": 12000,
                "height": 165.0,
                "weight": 70.0,
                "gender": 1,
                "ap_hi": 120,
                "ap_lo": 80,
                "cholesterol": 1,
                "gluc": 1,
                "cultural_belief_score": "Occasionally",
                "treatment_adherence": "High",
                "distance_to_healthcare": "Near"
            }
        }

# Retraining Data Model
class RetrainingData(BaseModel):
    patients: List[CardiovascularRiskInput]
    labels: List[int] = Field(..., description="Risk labels (0: Low Risk, 1: Medium Risk, 2: High Risk)")
    
    @validator('labels')
    def validate_labels(cls, v, values):
        if 'patients' in values and len(values['patients']) != len(v):
            raise ValueError("Number of patients and labels must match")
        if not all(label in [0, 1, 2] for label in v):
            raise ValueError("Labels must be 0 (Low Risk), 1 (Medium Risk), or 2 (High Risk)")
        return v

# Create FastAPI app
app = FastAPI(
    title="Cardiovascular Risk Prediction API", 
    description="Advanced API for cardiovascular risk prediction, model management, and data insights",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware for error handling and logging
@app.middleware("http")
async def log_requests(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Helper functions
def find_data_file(file_name):
    """
    Find a data file in various possible locations
    
    Args:
        file_name (str): File name to search for
        
    Returns:
        str: Path to the file if found
        
    Raises:
        FileNotFoundError: If file not found in any expected location
    """
    # List of possible paths
    possible_paths = [
        os.path.join('../data', file_name),
        os.path.join('data', file_name),
        os.path.join('./data', file_name),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', file_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', file_name),
        os.path.join(UPLOADS_DIR, file_name)
    ]
    
    # Try each path
    for path in possible_paths:
        if os.path.isfile(path):
            logger.info(f"Found data file at: {path}")
            return path
    
    # If we get here, file was not found
    raise FileNotFoundError(f"Could not find {file_name} in any expected location")

def _interpret_risk(risk_prob):
    """
    Provide a textual interpretation of risk probability
    
    Args:
        risk_prob (float): Predicted risk probability
    
    Returns:
        str: Risk interpretation
    """
    if risk_prob < 0.2:
        return "Low Risk: Cardiovascular health appears to be good."
    elif risk_prob < 0.4:
        return "Moderate-Low Risk: Some potential cardiovascular concerns."
    elif risk_prob < 0.6:
        return "Moderate Risk: Notable cardiovascular risk factors detected."
    elif risk_prob < 0.8:
        return "High-Moderate Risk: Significant cardiovascular risk observed."
    else:
        return "High Risk: Substantial cardiovascular risk, immediate medical consultation recommended."

# API Routes
@app.get("/")
async def root():
    """API health check and welcome endpoint"""
    return {
        "status": "operational",
        "api_version": "2.0.0",
        "documentation": "/docs"
    }

@app.post("/predict")
async def predict(
    patient: CardiovascularRiskInput, 
    save_to_db: Optional[bool] = False,
    background_tasks: BackgroundTasks = None
):
    """
    Predict cardiovascular risk for a single patient
    
    Args:
        patient (CardiovascularRiskInput): Patient data
        save_to_db (bool, optional): Whether to save prediction to database
        background_tasks (BackgroundTasks): FastAPI background tasks
    
    Returns:
        dict: Prediction results
    """
    logger.info("Processing prediction request")
    
    try:
        # Load the latest model
        try:
            model = load_latest_model()
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.error("No trained model found")
            raise HTTPException(
                status_code=404, 
                detail="No trained model found. Please train a model first."
            )
        
        # Prepare input data for preprocessing
        input_data = patient.dict()
        
        # Preprocess input
        try:
            input_scaled = preprocess_single_datapoint(input_data)
            logger.info("Data preprocessed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error preprocessing input data: {str(e)}"
            )
        
        # Predict
        try:
            risk_prob = model.predict(input_scaled)[0][0]
            risk_level = 2 if risk_prob > 0.7 else (1 if risk_prob > 0.3 else 0)
            logger.info(f"Prediction successful: risk_prob={risk_prob}, risk_level={risk_level}")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error during prediction: {str(e)}"
            )
        
        # Optionally save to database
        if save_to_db and background_tasks:
            # Add the predicted risk level to the input data
            input_data['risk_level'] = risk_level
            background_tasks.add_task(add_single_record, input_data)
            logger.info("Added background task to save prediction to database")
        
        return {
            "risk_probability": float(risk_prob),
            "risk_level": int(risk_level),
            "risk_category": ["Low", "Medium", "High"][risk_level],
            "interpretation": _interpret_risk(risk_prob)
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/retrain")
async def retrain_model(data: RetrainingData):
    """
    Retrain the model with new patient data, preserving all features
    
    Args:
        data (RetrainingData): Training data including patients and labels
    
    Returns:
        dict: Retraining results with accuracy and loss
    """
    logger.info(f"Processing retraining request with {len(data.patients)} new samples")
    
    try:
        # Convert the input data to a DataFrame
        patient_records = []
        for patient, label in zip(data.patients, data.labels):
            # Convert patient to dictionary and add label
            patient_dict = patient.dict()
            patient_dict['risk_level'] = label
            patient_records.append(patient_dict)
        
        # Create a DataFrame from the new records
        new_data_df = pd.DataFrame(patient_records)
        
        # Try to find the existing training data
        try:
            data_path = find_data_file('cardio_train.csv')
            logger.info(f"Using existing data from: {data_path}")
            
            # Load and process both datasets with drop_first=False to preserve all features
            X_existing, Y_existing = load_and_preprocess_data(data_path, is_file=True, drop_first=False)
            logger.info(f"Existing data processed with shape: {X_existing.shape}")
            
            X_new, Y_new = load_and_preprocess_data(new_data_df, is_file=False, drop_first=False)
            logger.info(f"New data processed with shape: {X_new.shape}")
            
            # Ensure column consistency between datasets
            all_columns = set(X_existing.columns) | set(X_new.columns)
            
            # Add any missing columns to both datasets
            for col in all_columns:
                if col not in X_existing.columns:
                    logger.info(f"Adding missing column {col} to existing data")
                    X_existing[col] = 0
                if col not in X_new.columns:
                    logger.info(f"Adding missing column {col} to new data")
                    X_new[col] = 0
            
            # Ensure column order is the same
            feature_columns = sorted(list(all_columns))
            
            # Reorder columns for consistency
            X_existing = X_existing[feature_columns]
            X_new = X_new[feature_columns]
            
            # Combine datasets
            X_combined = pd.concat([X_existing, X_new], axis=0)
            Y_combined = pd.concat([Y_existing, Y_new], axis=0)
            
            logger.info(f"Combined data shape: {X_combined.shape}")
            logger.info(f"Feature count: {len(feature_columns)}")
            
            # Perform scaling and splitting
            X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler = scale_and_split_data(
                X_combined, Y_combined
            )
            logger.info("Data scaled and split successfully")
            
            # Build a new model that can handle all features
            input_feature_count = X_train.shape[1]
            logger.info(f"Building new model with input shape: {input_feature_count}")
            
            new_model = build_new_model(input_feature_count)
            
            # Train the new model
            fine_tuned_model, history = fine_tune_model(
                new_model,
                X_train,
                Y_train,
                X_val,
                Y_val,
                is_new_model=True
            )
            logger.info("Model training completed successfully")
            
            # Evaluate the model
            loss, accuracy = fine_tuned_model.evaluate(X_test, Y_test, verbose=0)
            logger.info(f"Model evaluation: loss={loss}, accuracy={accuracy}")
            
            # Add the new records to the database
            try:
                for record in patient_records:
                    add_single_record(record)
                logger.info(f"Added {len(patient_records)} new records to database")
            except Exception as e:
                logger.warning(f"Error adding records to database (model still retrained): {e}")
            
            return {
                "message": "Model retrained successfully with all features preserved",
                "num_samples": len(data.patients),
                "accuracy": float(accuracy),
                "loss": float(loss),
                "feature_count": input_feature_count,
                "history": {
                    "loss": [float(val) for val in history.history.get('loss', [])],
                    "val_loss": [float(val) for val in history.history.get('val_loss', [])],
                    "accuracy": [float(val) for val in history.history.get('accuracy', [])],
                    "val_accuracy": [float(val) for val in history.history.get('val_accuracy', [])]
                }
            }
            
        except FileNotFoundError:
            logger.error("No existing training data found")
            raise HTTPException(
                status_code=404, 
                detail="No existing training data found. Please upload training data first."
            )
            
    except ValueError as e:
        logger.error(f"Value error during retraining: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during retraining: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/visualize/{feature_name}")
async def visualize_feature(feature_name: str):
    """
    Generate visualization for a specific feature
    
    Args:
        feature_name (str): Name of the feature to visualize
    
    Returns:
        dict: Visualization data
    """
    try:
        # Find data file
        data_path = find_data_file('cardio_train.csv')
        
        # Generate visualization
        img_str = generate_visualization(feature_name, data_path)
        
        return {"feature": feature_name, "image": img_str}
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Visualization error for {feature_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize/feature_importance")
async def visualize_feature_importance():
    """
    Generate feature importance visualization
    
    Returns:
        dict: Feature importance visualization
    """
    try:
        # Find data file
        data_path = find_data_file('cardio_train.csv')
        
        # Generate feature importance
        img_str = generate_feature_importance(data_path)
        
        return {"image": img_str}
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Feature importance visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload CSV data file for training or prediction
    
    Args:
        file (UploadFile): CSV file to upload
    
    Returns:
        dict: Upload results
    """
    try:
        # Validate file extension
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV files are supported"
            )
        
        # Create uploads directory if it doesn't exist
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        # Save uploaded file with secure filename
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        
        # Write file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to: {file_path}")
        
        # Try to import to database
        try:
            records_imported = import_csv_to_db(file_path)
            logger.info(f"Imported {records_imported} records to database")
        except Exception as e:
            logger.warning(f"Database import failed: {e}")
            return {
                "filename": file.filename,
                "path": file_path,
                "status": "File uploaded successfully, but database import failed",
                "error": str(e)
            }
        
        return {
            "filename": file.filename,
            "path": file_path,
            "records_imported": records_imported,
            "status": "File uploaded and imported successfully"
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Data upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    API health check endpoint
    
    Returns:
        dict: Health status information
    """
    # Check for model availability
    model_available = False
    try:
        load_latest_model()
        model_available = True
    except:
        pass
    
    # Check for training data availability
    data_available = False
    try:
        find_data_file('cardio_train.csv')
        data_available = True
    except:
        pass
    
    return {
        "status": "healthy",
        "model_available": model_available,
        "training_data_available": data_available,
        "api_version": "2.0.0"
    }

# Main entry point
if __name__ == "__main__":
    try:
        # Check if models directory exists
        if not os.path.exists(MODELS_DIR):
            logger.warning(f"Models directory not found. Creating: {MODELS_DIR}")
            os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Check if uploads directory exists
        if not os.path.exists(UPLOADS_DIR):
            logger.warning(f"Uploads directory not found. Creating: {UPLOADS_DIR}")
            os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        logger.info("Starting Cardiovascular Risk Prediction API server")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.critical(f"Failed to start server: {e}")
        logger.critical(traceback.format_exc())
