import pandas as pd
import numpy as np
from datetime import datetime
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import ssl
import certifi
import os
import sys  # Added import

# MongoDB connection settings
DB_NAME = "cardio_database"
COLLECTION_NAME = "cardio_info"
BATCH_COLLECTION = "upload_batches"

def get_mongo_client():
    try:
        # MongoDB connection string
        MONGO_URI = "mongodb+srv://jssozi:J0788565007ynn@ac-ti8dfyk.tzizffo.mongodb.net/cardio_database?retryWrites=true&w=majority"
        
        # Simplified connection options with minimal parameters
        client = MongoClient(
            MONGO_URI,
            server_api=ServerApi('1'),
            # Only use essential TLS parameters
            tlsCAFile=certifi.where(),
            # Reduced timeouts for faster failure detection
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        
        # Test connection with a lighter command
        client.admin.command('ismaster')
        print("Successfully connected to MongoDB!")
        return client
    
    except Exception as e:
        print(f"Unexpected error connecting to MongoDB: {e}")
        # Print diagnostic information
        print(f"Python version: {sys.version}")
        print(f"PyMongo version: {pymongo.__version__}")
        print(f"SSL/TLS version: {ssl.OPENSSL_VERSION}")
        
        # Try DNS lookup to verify connectivity
        try:
            import socket
            hostname = "ac-ti8dfyk.tzizffo.mongodb.net"
            ip_addresses = socket.gethostbyname_ex(hostname)
            print(f"DNS lookup for {hostname}: {ip_addresses}")
        except Exception as dns_error:
            print(f"DNS lookup failed: {dns_error}")
            
        raise

def debug_ssl_connection():
    """
    Comprehensive SSL and network debugging function
    """
    import socket
    import ssl
    import certifi

    try:
        # Hostname and port from your MongoDB connection string
        hostname = 'ac-ti8dfyk.tzizffo.mongodb.net'
        port = 27017

        # Create a socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)  # 30-second timeout

        # Create SSL context
        context = ssl.create_default_context(cafile=certifi.where())
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED

        # Wrap socket with SSL
        secure_sock = context.wrap_socket(sock, server_hostname=hostname)

        try:
            # Attempt to connect
            secure_sock.connect((hostname, port))
            print("SSL Connection successful!")
            
            # Get certificate details
            cert = secure_sock.getpeercert()
            print("Certificate details:")
            for key, value in cert.items():
                print(f"{key}: {value}")
        
        except ssl.SSLError as ssl_err:
            print(f"SSL Error during connection: {ssl_err}")
        except socket.error as sock_err:
            print(f"Socket Error: {sock_err}")
        finally:
            secure_sock.close()

    except Exception as e:
        print(f"Debugging Error: {e}")

def ensure_db_exists():
    """Ensure indexes and connections are properly set up"""
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        
        # Create indexes for better query performance
        db[COLLECTION_NAME].create_index("used_for_training", background=True)
        db[COLLECTION_NAME].create_index("risk_level", background=True)
        db[COLLECTION_NAME].create_index("upload_date", background=True)
        
        # Close the connection
        client.close()
    except Exception as e:
        print(f"Error in ensure_db_exists: {e}")
        raise

def import_csv_to_db(csv_path, delimiter=";"):
    """
    Import a CSV file into the MongoDB database, using the existing preprocessing function
    
    Args:
        csv_path (str): Path to the CSV file
        delimiter (str): CSV delimiter
    
    Returns:
        int: Number of records imported
    """
    # Use your existing preprocessing function to load and preprocess data
    X, Y = load_and_preprocess_data(csv_path)
    
    # Combine X and Y back into a single DataFrame with all processed features
    data = X.copy()
    data['risk_level'] = Y
    
    # Add upload date and training flag
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['upload_date'] = current_date
    data['used_for_training'] = False
    
    # Convert to list of dictionaries for MongoDB
    records = data.to_dict('records')
    
    # Connect to MongoDB and insert the data
    client = get_mongo_client()
    db = client[DB_NAME]
    
    # Insert the data
    result = db[COLLECTION_NAME].insert_many(records)
    
    # Record the upload batch
    db[BATCH_COLLECTION].insert_one({
        "filename": os.path.basename(csv_path),
        "upload_date": current_date,
        "num_records": len(records)
    })
    
    client.close()
    
    return len(records)

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
    # Create a pandas DataFrame from the record_data
    df = pd.DataFrame([record_data])
    
    # Use the preprocess_single_datapoint function which should handle all transformations
    processed_data = preprocess_single_datapoint(record_data)
    
    # The preprocessing function might return a numpy array, DataFrame or dict
    # Handle each case appropriately
    if isinstance(processed_data, np.ndarray):
        # Convert numpy array to dict using original record keys for structure
        # This is a simplified approach - you might need to adjust based on your preprocessing
        mongo_record = record_data.copy()
        # Add preprocessed risk level if available
        if hasattr(processed_data, 'shape') and len(processed_data.shape) > 1:
            mongo_record['risk_level'] = int(np.argmax(processed_data[0]))
    elif isinstance(processed_data, pd.DataFrame):
        # Convert DataFrame to dict
        mongo_record = processed_data.to_dict('records')[0]
    else:
        # Already a dict-like structure
        mongo_record = processed_data
    
    # Add metadata fields
    mongo_record['upload_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mongo_record['used_for_training'] = False
    
    # Connect to MongoDB and insert
    client = get_mongo_client()
    db = client[DB_NAME]
    
    # Insert the record
    result = db[COLLECTION_NAME].insert_one(mongo_record)
    
    client.close()
    
    return str(result.inserted_id)

# Initialize database connection when the module is imported
ensure_db_exists()
