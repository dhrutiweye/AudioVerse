"""Database package for MongoDB operations."""
from .mongo_client import (
    MongoClientManager,
    get_mongo_client,
    get_database,
    close_mongo_client,
    get_document_by_id,
    get_documents_by_date_range,
    get_documents_by_filter,
    get_unique_values_by_filter,
    insert_documents,
    write_in_mongo,
    update_document,
    delete_document,
)

__all__ = [
    "MongoClientManager",
    "get_mongo_client",
    "get_database",
    "close_mongo_client",
    "get_document_by_id",
    "get_documents_by_date_range",
    "get_documents_by_filter",
    "get_unique_values_by_filter",
    "insert_documents",
    "write_in_mongo",
    "update_document",
    "delete_document",
]

