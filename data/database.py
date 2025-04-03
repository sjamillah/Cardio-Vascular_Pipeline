import pandas as pd
import numpy as np
from datetime import datetime
import pymongo
from pymongo.mongo_client import MongoClient
import ssl
import certifi
import os
import sys
from src.preprocessing import load_and_preprocess_data, preprocess_single_datapoint

# MongoDB connection settings
DB_NAME = "cardio_database"
COLLECTION_NAME = "cardio_info"
BATCH_COLLECTION = "upload_batches"

# Modified get_mongo_client function with fallback options
def get_mongo_client():
    try:
        srv_uri = "mongodb+srv://jssozi:J0788565007ynn@ac-ti8dfyk.tzizffo.mongodb.net/cardio_database?retryWrites=true&w=majority"
        
        # Try with minimal SSL options and allow invalid certificates for testing
        client = MongoClient(
            srv_uri,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=False,
            ssl_cert_reqs=ssl.CERT_REQUIRED,
            ssl_min_tls_version=ssl.PROTOCOL_TLSv1_2,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=30000
        )
        
        # Simple ping test - no additional authentication required
        client.admin.command('ping')
        print(f"Connected to MongoDB! OpenSSL: {ssl.OPENSSL_VERSION}")
        return client
        
    except Exception as e:
        print(f"Connection failed: {e}")
        print(f"Python: {sys.version}")
        print(f"PyMongo: {pymongo.__version__}")
        print(f"OpenSSL: {ssl.OPENSSL_VERSION}")
        print("WARNING: MongoDB connection failed. Running with limited functionality.")
        return None

# Modified ensure_db_exists function to handle MongoDB connection failure
def ensure_db_exists():
    """Ensure indexes and connections are properly set up"""
    try:
        client = get_mongo_client()
        
        # If MongoDB connection failed but we want app to start anyway
        if client is None:
            print("Skipping database initialization - MongoDB unavailable")
            return False
            
        db = client[DB_NAME]
        
        # Create indexes for better query performance
        db[COLLECTION_NAME].create_index("used_for_training", background=True)
        db[COLLECTION_NAME].create_index("risk_level", background=True)
        db[COLLECTION_NAME].create_index("upload_date", background=True)
        
        # Close the connection
        client.close()
        return True
        
    except Exception as e:
        print(f"Error in ensure_db_exists: {e}")
        print("Warning: Application will start with limited database functionality")
        return False

def import_csv_to_db(csv_path, delimiter=";"):
    """
    Import a CSV file into the MongoDB database, using the existing preprocessing function
    
    Args:
        csv_path (str): Path to the CSV file
        delimiter (str): CSV delimiter
    
    Returns:
        int: Number of records imported
    """
    client = get_mongo_client()
    if client is None:
        print("MongoDB unavailable, skipping CSV import")
        return 0
    try:
        X, Y = load_and_preprocess_data(csv_path)
        data = X.copy()
        data['risk_level'] = Y
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data['upload_date'] = current_date
        data['used_for_training'] = False
        records = data.to_dict('records')
        
        db = client[DB_NAME]
        result = db[COLLECTION_NAME].insert_many(records)
        db[BATCH_COLLECTION].insert_one({
            "filename": os.path.basename(csv_path),
            "upload_date": current_date,
            "num_records": len(records)
        })
        client.close()
        return len(records)
    except Exception as e:
        print(f"Error importing CSV: {e}")
        return 0

def get_training_data(limit=None, only_new=True):
    """
    Get data for model training
    
    Args:
        limit (int): Maximum number of records to retrieve
        only_new (bool): If True, only get records not used for training yet
    
    Returns:
        tuple: (X, Y) - Features and target variable
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    
    # Construct the query
    query = {}
    if only_new:
        query = {"used_for_training": False}
    
    # Get the data
    cursor = db[COLLECTION_NAME].find(query)
    if limit:
        cursor = cursor.limit(limit)
    
    # Convert to DataFrame
    data = pd.DataFrame(list(cursor))
    
    if len(data) == 0:
        client.close()
        return None, None
    
    # Mark records as used for training
    if only_new and len(data) > 0:
        record_ids = [str(r["_id"]) for r in data.to_dict('records')]
        db[COLLECTION_NAME].update_many(
            {"_id": {"$in": [r for r in data["_id"]]}},
            {"$set": {"used_for_training": True}}
        )
    
    # Remove MongoDB _id field and other non-feature fields
    if '_id' in data.columns:
        data = data.drop('_id', axis=1)
    
    # The data is already preprocessed, just need to separate X and Y
    non_feature_cols = ['upload_date', 'used_for_training']
    data_clean = data.drop(non_feature_cols, axis=1)
    
    X = data_clean.drop('risk_level', axis=1)
    Y = data_clean['risk_level']
    
    client.close()
    
    return X, Y

def get_record_count():
    """
    Get the number of records in the database
    
    Returns:
        int: Number of records
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    
    count = db[COLLECTION_NAME].count_documents({})
    
    client.close()
    
    return count

def get_training_stats():
    """
    Get statistics about training data
    
    Returns:
        dict: Statistics about the data
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    
    # Get total count
    total_count = db[COLLECTION_NAME].count_documents({})
    
    # Get count of records used for training
    used_for_training = db[COLLECTION_NAME].count_documents({"used_for_training": True})
    
    # Get count by risk level
    pipeline = [
        {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    risk_level_results = list(db[COLLECTION_NAME].aggregate(pipeline))
    risk_level_counts = {f"risk_level_{row['_id']}": row['count'] for row in risk_level_results}
    
    # Get recent upload batches
    recent_batches = list(db[BATCH_COLLECTION].find().sort("upload_date", -1).limit(5))
    for batch in recent_batches:
        batch["batch_id"] = str(batch["_id"])
        del batch["_id"]
    
    client.close()
    
    return {
        "total_records": total_count,
        "used_for_training": used_for_training,
        "unused_for_training": total_count - used_for_training,
        "risk_level_counts": risk_level_counts,
        "recent_batches": recent_batches
    }

def export_to_csv(output_path, limit=None):
    """
    Export data from the database to a CSV file
    
    Args:
        output_path (str): Path to save the CSV file
        limit (int): Maximum number of records to export
    
    Returns:
        int: Number of records exported
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    
    # Get the data
    cursor = db[COLLECTION_NAME].find({})
    if limit:
        cursor = cursor.limit(limit)
    
    # Convert to DataFrame
    data = pd.DataFrame(list(cursor))
    
    if len(data) == 0:
        client.close()
        return 0
    
    # Remove MongoDB _id field
    if '_id' in data.columns:
        data = data.drop('_id', axis=1)
    
    # Save to CSV
    data.to_csv(output_path, index=False)
    
    client.close()
    
    return len(data)

def add_single_record(record_data):
    """
    Add a single record to the database
    
    Args:
        record_data (dict): Record data
    
    Returns:
        str: ID of the inserted record
    """
    client = get_mongo_client()
    if client is None:
        print("MongoDB unavailable, skipping record insertion")
        return None
    try:
        processed_data = preprocess_single_datapoint(record_data)
        if isinstance(processed_data, np.ndarray):
            mongo_record = record_data.copy()
            mongo_record['risk_level'] = int(np.argmax(processed_data[0])) if processed_data.shape[1] > 1 else int(processed_data[0])
        elif isinstance(processed_data, pd.DataFrame):
            mongo_record = processed_data.to_dict('records')[0]
        else:
            mongo_record = processed_data
        
        mongo_record['upload_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mongo_record['used_for_training'] = False
        
        db = client[DB_NAME]
        result = db[COLLECTION_NAME].insert_one(mongo_record)
        client.close()
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error adding record: {e}")
        return None

# Initialize database connection when the module is imported
ensure_db_exists()
