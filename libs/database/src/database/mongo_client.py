"""MongoDB client that reads connection URL and database name from environment variables."""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from pymongo import MongoClient
from pymongo.database import Database


class MongoClientManager:
    """Manages MongoDB client connection with configuration from environment variables.
    
    Environment variables:
        MONGO_URL: MongoDB connection URL (e.g., 'mongodb://localhost:27017' or 
                   'mongodb://user:password@host:port/')
        MONGO_DB_NAME: Name of the database to connect to
    
    Example:
        >>> from database import get_mongo_client, get_database
        >>> client = get_mongo_client()
        >>> db = get_database()
    """
    
    _client: Optional[MongoClient] = None
    _database: Optional[Database] = None
    
    @classmethod
    def get_client(cls) -> MongoClient:
        """Get or create MongoDB client instance.
        
        Reads MONGO_URL from environment variable. If not set, defaults to
        'mongodb://localhost:27017'.
        
        Returns:
            MongoClient instance
            
        Raises:
            ValueError: If MONGO_URL is invalid or connection fails
        """
        if cls._client is None:
            mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
            
            if not mongo_url:
                raise ValueError("MONGO_URL environment variable is required")
            
            try:
                cls._client = MongoClient(mongo_url)
                # Test the connection
                cls._client.admin.command('ping')
            except Exception as e:
                raise ValueError(f"Failed to connect to MongoDB at {mongo_url}: {str(e)}")
        
        return cls._client
    
    @classmethod
    def get_database(cls) -> Database:
        """Get or create database instance.
        
        Reads MONGO_DB_NAME from environment variable. If not set, raises ValueError.
        
        Returns:
            Database instance
            
        Raises:
            ValueError: If MONGO_DB_NAME is not set or client connection fails
        """
        if cls._database is None:
            db_name = os.getenv("MONGO_DB_NAME")
            
            if not db_name:
                raise ValueError("MONGO_DB_NAME environment variable is required")
            
            client = cls.get_client()
            cls._database = client[db_name]
        
        return cls._database
    
    @classmethod
    def close_client(cls) -> None:
        """Close the MongoDB client connection."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            cls._database = None
    
    @classmethod
    def reset(cls) -> None:
        """Reset the client and database instances (useful for testing)."""
        cls.close_client()


def get_mongo_client() -> MongoClient:
    """Convenience function to get MongoDB client.
    
    Returns:
        MongoClient instance
    """
    return MongoClientManager.get_client()


def get_database() -> Database:
    """Convenience function to get database instance.
    
    Returns:
        Database instance
    """
    return MongoClientManager.get_database()


def close_mongo_client() -> None:
    """Convenience function to close MongoDB client connection."""
    MongoClientManager.close_client()


# Default MongoDB URL from environment variable
def _get_default_mongo_url() -> str:
    """Get default MongoDB URL from environment variable."""
    return os.getenv("MONGO_URL", "mongodb://localhost:27017")


def get_document_by_id(
    db_name: str,
    collection_name: str,
    doc_id: Any,
    uri: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch a document from MongoDB by collection name and ID.

    Args:
        db_name (str): The database name
        collection_name (str): The collection name
        doc_id: The document ID (can be string, int, ObjectId, etc.)
        uri (str, optional): MongoDB connection string. If not provided, uses MONGO_URL env var

    Returns:
        dict | None: The document if found, otherwise None
    """
    mongo_url = uri or _get_default_mongo_url()
    client = None
    
    try:
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        document = collection.find_one({"_id": doc_id})
        return document

    except Exception as e:
        print(f"Error fetching document: {e}")
        return None
    
    finally:
        if client:
            client.close()


def get_documents_by_date_range(
    db_name: str,
    collection_name: str,
    start_date: datetime,
    end_date: datetime,
    limit: int = 50,
    offset: int = 0,
    uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch documents from MongoDB within a given date range.

    Args:
        db_name (str): The database name
        collection_name (str): The collection name
        start_date (datetime): Start datetime (inclusive)
        end_date (datetime): End datetime (exclusive)
        limit (int): Maximum number of documents to return. Defaults to 50.
        offset (int): Number of documents to skip. Defaults to 0.
        uri (str, optional): MongoDB connection string. If not provided, uses MONGO_URL env var

    Returns:
        list[dict]: List of matching documents
    """
    mongo_url = uri or _get_default_mongo_url()
    client = None
    print(f"connecting {mongo_url}")
    
    try:
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]
        print(f"collection {collection}")

        cursor = (
            collection.find({
                "timestamp": {
                    "$gte": start_date,
                    "$lt": end_date
                }
            })
            .skip(offset)
            .limit(limit)
        )

        return list(cursor)

    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []
    
    finally:
        if client:
            client.close()


def get_documents_by_filter(
    db_name: str,
    collection_name: str,
    data: Dict[str, Any],
    uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch documents from MongoDB using a filter.

    Args:
        db_name (str): The database name
        collection_name (str): The collection name
        data (dict): Filter query dictionary
        uri (str, optional): MongoDB connection string. If not provided, uses MONGO_URL env var

    Returns:
        list[dict]: List of matching documents
    """
    mongo_url = uri or _get_default_mongo_url()
    client = None
    
    try:
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        cursor = collection.find(data)
        return list(cursor)

    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []
    
    finally:
        if client:
            client.close()


def get_unique_values_by_filter(
    db_name: str,
    collection_name: str,
    column_name: str,
    filter_data: Optional[Dict[str, Any]] = None,
    offset: int = 0,
    limit: int = 50,
    uri: Optional[str] = None,
) -> List[Any]:
    """
    Fetch unique values for a given column from a MongoDB collection.

    Args:
        db_name (str): MongoDB database name.
        collection_name (str): Collection name.
        column_name (str): Field name for which to get distinct values.
        filter_data (dict, optional): Filter query (if None, retrieves all documents).
        offset (int, optional): Number of results to skip. Defaults to 0.
        limit (int, optional): Maximum number of results to return. Defaults to 50.
        uri (str, optional): MongoDB URI. If not provided, uses MONGO_URL env var.

    Returns:
        List[Any]: List of unique values for the given column.
    """
    mongo_url = uri or _get_default_mongo_url()
    client = None
    print(f"connecting to {mongo_url}")
    try:
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        # If no filter provided, use empty dict
        filter_query = filter_data if filter_data else {}

        # Get distinct values
        unique_values = collection.distinct(column_name, filter_query)

        # Apply offset and limit
        paginated_values = unique_values[offset:offset + limit]

        return paginated_values

    except Exception as e:
        print(f"Error fetching unique values: {e}")
        return []
    finally:
        if client:
            client.close()


def insert_documents(
    db_name: str,
    collection_name: str,
    data_list: List[Dict[str, Any]],
    uri: Optional[str] = None,
) -> List[str]:
    """
    Insert multiple documents into a MongoDB collection.

    Args:
        db_name (str): The database name
        collection_name (str): The collection name
        data_list (list[dict]): List of documents to insert
        uri (str, optional): MongoDB connection string. If not provided, uses MONGO_URL env var

    Returns:
        list[str]: List of inserted document IDs as strings
    """
    mongo_url = uri or _get_default_mongo_url()
    client = None
    
    try:
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        result = collection.insert_many(data_list)
        return [str(_id) for _id in result.inserted_ids]

    except Exception as e:
        print(f"Error inserting documents: {e}")
        return []
    
    finally:
        if client:
            client.close()


def write_in_mongo(
    db_name: str,
    collection_name: str,
    data: Dict[str, Any],
    uri: Optional[str] = None,
) -> Optional[str]:
    """
    Insert a single document into a MongoDB collection.

    Args:
        db_name (str): The database name
        collection_name (str): The collection name
        data (dict): Document to insert
        uri (str, optional): MongoDB connection string. If not provided, uses MONGO_URL env var

    Returns:
        str | None: The inserted document ID as string, or None on error
    """
    mongo_url = uri or _get_default_mongo_url()
    client = None
    
    try:
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        result = collection.insert_one(data)
        return str(result.inserted_id)

    except Exception as e:
        print(f"Error inserting document: {e}")
        return None
    
    finally:
        if client:
            client.close()


def update_document(
    db_name: str,
    collection_name: str,
    doc_id: Any,
    updates: Dict[str, Any],
    uri: Optional[str] = None,
) -> int:
    """
    Update a document in MongoDB by ID.

    Args:
        db_name (str): The database name
        collection_name (str): The collection name
        doc_id: The document ID to update
        updates (dict): Dictionary of fields to update
        uri (str, optional): MongoDB connection string. If not provided, uses MONGO_URL env var

    Returns:
        int: Number of documents modified (0 or 1)
        
    Raises:
        ValueError: If updates dictionary is empty
    """
    if not updates:
        raise ValueError("No updates provided.")
    
    mongo_url = uri or _get_default_mongo_url()
    client = None

    try:
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        result = collection.update_one({"_id": doc_id}, {"$set": updates}, upsert=False)
        return result.modified_count

    except Exception as e:
        print(f"Error updating document: {e}")
        return 0
    
    finally:
        if client:
            client.close()


def delete_document(
    db_name: str,
    collection_name: str,
    doc_id: Any,
    uri: Optional[str] = None,
) -> int:
    """
    Delete a document from MongoDB by ID.

    Args:
        db_name (str): The database name
        collection_name (str): The collection name
        doc_id: The document ID to delete
        uri (str, optional): MongoDB connection string. If not provided, uses MONGO_URL env var

    Returns:
        int: Number of documents deleted (0 or 1)
    """
    mongo_url = uri or _get_default_mongo_url()
    client = None
    
    try:
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        result = collection.delete_one({"_id": doc_id})
        return result.deleted_count

    except Exception as e:
        print(f"Error deleting document: {e}")
        return 0
    
    finally:
        if client:
            client.close()

