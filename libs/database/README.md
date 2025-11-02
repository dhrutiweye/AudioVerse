# Database Library

MongoDB client library with comprehensive CRUD operations and connection management.

## Setup

Set the following environment variables:

```bash
export MONGO_URL="mongodb://localhost:27017"
export MONGO_DB_NAME="my_database"
```

Or use a connection string with authentication:

```bash
export MONGO_URL="mongodb://username:password@host:27017/"
export MONGO_DB_NAME="my_database"
```

## Installation

```bash
poetry install
```

## Usage

### Connection Management

#### `get_mongo_client()`

Get a MongoDB client instance using the connection URL from environment variables.

```python
from database import get_mongo_client

# Uses MONGO_URL from environment or defaults to mongodb://localhost:27017
client = get_mongo_client()
print(f"Connected to: {client.server_info()}")
```

#### `get_database()`

Get a database instance using the database name from environment variables.

```python
from database import get_database

# Uses MONGO_DB_NAME from environment
db = get_database()
collection = db["my_collection"]
```

#### `close_mongo_client()`

Close the MongoDB client connection.

```python
from database import close_mongo_client

close_mongo_client()
```

#### `MongoClientManager`

Use the manager class directly for more control:

```python
from database import MongoClientManager

# Get client
client = MongoClientManager.get_client()

# Get database
db = MongoClientManager.get_database()

# Close connection
MongoClientManager.close_client()
```

---

## Read Operations

### `get_document_by_id()`

Fetch a single document by its ID.

```python
from database import get_document_by_id

# Using default URI from environment
doc = get_document_by_id(
    db_name="my_database",
    collection_name="users",
    doc_id="507f1f77bcf86cd799439011"
)

print(doc)  # {'_id': '507f1f77bcf86cd799439011', 'name': 'John', ...}

# Using custom URI
doc = get_document_by_id(
    db_name="my_database",
    collection_name="users",
    doc_id=123,
    uri="mongodb://localhost:27017"
)

# Returns None if document not found
if doc is None:
    print("Document not found")
```

### `get_documents_by_date_range()`

Fetch documents within a date range with pagination.

```python
from database import get_documents_by_date_range
from datetime import datetime

# Get documents from last week
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 31)

documents = get_documents_by_date_range(
    db_name="my_database",
    collection_name="logs",
    start_date=start_date,
    end_date=end_date,
    limit=50,
    offset=0
)

print(f"Found {len(documents)} documents")

# With pagination
page_1 = get_documents_by_date_range(
    db_name="my_database",
    collection_name="logs",
    start_date=start_date,
    end_date=end_date,
    limit=10,
    offset=0
)

page_2 = get_documents_by_date_range(
    db_name="my_database",
    collection_name="logs",
    start_date=start_date,
    end_date=end_date,
    limit=10,
    offset=10
)

# Using custom URI
documents = get_documents_by_date_range(
    db_name="my_database",
    collection_name="logs",
    start_date=start_date,
    end_date=end_date,
    uri="mongodb://localhost:27017"
)
```

**Note:** The method queries documents with a `timestamp` field. Make sure your documents have this field.

### `get_documents_by_filter()`

Fetch documents using a MongoDB filter query.

```python
from database import get_documents_by_filter

# Simple filter
documents = get_documents_by_filter(
    db_name="my_database",
    collection_name="users",
    data={"status": "active"}
)

print(f"Found {len(documents)} active users")

# Complex filter with multiple conditions
documents = get_documents_by_filter(
    db_name="my_database",
    collection_name="users",
    data={
        "status": "active",
        "age": {"$gte": 18},
        "city": {"$in": ["New York", "London"]}
    }
)

# Using MongoDB operators
documents = get_documents_by_filter(
    db_name="my_database",
    collection_name="products",
    data={
        "$or": [
            {"category": "electronics"},
            {"price": {"$lt": 100}}
        ]
    }
)

# Using custom URI
documents = get_documents_by_filter(
    db_name="my_database",
    collection_name="users",
    data={"status": "active"},
    uri="mongodb://localhost:27017"
)
```

### `get_unique_values_by_filter()`

Get distinct values for a specific column/field.

```python
from database import get_unique_values_by_filter

# Get all unique cities
cities = get_unique_values_by_filter(
    db_name="my_database",
    collection_name="users",
    column_name="city"
)

print(f"Unique cities: {cities}")

# Get unique values with filter
active_cities = get_unique_values_by_filter(
    db_name="my_database",
    collection_name="users",
    column_name="city",
    filter_data={"status": "active"}
)

# With pagination
first_10_cities = get_unique_values_by_filter(
    db_name="my_database",
    collection_name="users",
    column_name="city",
    offset=0,
    limit=10
)

next_10_cities = get_unique_values_by_filter(
    db_name="my_database",
    collection_name="users",
    column_name="city",
    offset=10,
    limit=10
)

# Using custom URI
values = get_unique_values_by_filter(
    db_name="my_database",
    collection_name="users",
    column_name="email",
    uri="mongodb://localhost:27017"
)
```

---

## Write Operations

### `write_in_mongo()`

Insert a single document into a collection.

```python
from database import write_in_mongo

# Insert a single document
document = {
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30,
    "city": "New York"
}

doc_id = write_in_mongo(
    db_name="my_database",
    collection_name="users",
    data=document
)

print(f"Document inserted with ID: {doc_id}")

# Using custom URI
doc_id = write_in_mongo(
    db_name="my_database",
    collection_name="users",
    data=document,
    uri="mongodb://localhost:27017"
)

# Returns None on error
if doc_id is None:
    print("Failed to insert document")
```

### `insert_documents()`

Insert multiple documents at once (bulk insert).

```python
from database import insert_documents

# Insert multiple documents
documents = [
    {"name": "Alice", "email": "alice@example.com", "age": 25},
    {"name": "Bob", "email": "bob@example.com", "age": 28},
    {"name": "Charlie", "email": "charlie@example.com", "age": 32}
]

inserted_ids = insert_documents(
    db_name="my_database",
    collection_name="users",
    data_list=documents
)

print(f"Inserted {len(inserted_ids)} documents")
print(f"Document IDs: {inserted_ids}")

# Using custom URI
inserted_ids = insert_documents(
    db_name="my_database",
    collection_name="users",
    data_list=documents,
    uri="mongodb://localhost:27017"
)

# Returns empty list on error
if not inserted_ids:
    print("Failed to insert documents")
```

---

## Update Operations

### `update_document()`

Update a document by ID.

```python
from database import update_document

# Update document fields
updates = {
    "age": 31,
    "city": "Boston",
    "status": "premium"
}

modified_count = update_document(
    db_name="my_database",
    collection_name="users",
    doc_id="507f1f77bcf86cd799439011",
    updates=updates
)

print(f"Modified {modified_count} document(s)")

# Using integer ID
modified_count = update_document(
    db_name="my_database",
    collection_name="users",
    doc_id=123,
    updates={"age": 35}
)

# Using custom URI
modified_count = update_document(
    db_name="my_database",
    collection_name="users",
    doc_id="507f1f77bcf86cd799439011",
    updates={"status": "active"},
    uri="mongodb://localhost:27017"
)

# Raises ValueError if updates is empty
try:
    update_document(
        db_name="my_database",
        collection_name="users",
        doc_id="507f1f77bcf86cd799439011",
        updates={}  # This will raise ValueError
    )
except ValueError as e:
    print(f"Error: {e}")
```

**Note:** This method uses `$set` operator for updates. Returns 0 if document not found or update fails.

---

## Delete Operations

### `delete_document()`

Delete a document by ID.

```python
from database import delete_document

# Delete a document
deleted_count = delete_document(
    db_name="my_database",
    collection_name="users",
    doc_id="507f1f77bcf86cd799439011"
)

print(f"Deleted {deleted_count} document(s)")

# Using integer ID
deleted_count = delete_document(
    db_name="my_database",
    collection_name="users",
    doc_id=123
)

# Using custom URI
deleted_count = delete_document(
    db_name="my_database",
    collection_name="users",
    doc_id="507f1f77bcf86cd799439011",
    uri="mongodb://localhost:27017"
)

# Returns 0 if document not found or delete fails
if deleted_count == 0:
    print("Document not found or deletion failed")
```

---

## Complete Example

Here's a complete workflow example:

```python
from database import (
    write_in_mongo,
    get_document_by_id,
    update_document,
    delete_document,
    get_documents_by_filter
)

# 1. Insert a document
doc_id = write_in_mongo(
    db_name="my_database",
    collection_name="users",
    data={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30
    }
)
print(f"Created user with ID: {doc_id}")

# 2. Read the document
user = get_document_by_id(
    db_name="my_database",
    collection_name="users",
    doc_id=doc_id
)
print(f"User: {user}")

# 3. Update the document
update_document(
    db_name="my_database",
    collection_name="users",
    doc_id=doc_id,
    updates={"age": 31, "status": "active"}
)

# 4. Query multiple documents
active_users = get_documents_by_filter(
    db_name="my_database",
    collection_name="users",
    data={"status": "active"}
)
print(f"Active users: {len(active_users)}")

# 5. Delete the document
deleted_count = delete_document(
    db_name="my_database",
    collection_name="users",
    doc_id=doc_id
)
print(f"Deleted {deleted_count} document(s)")
```

---

## Error Handling

All methods handle errors gracefully:

- **Read operations** return `None` (for single document) or `[]` (for lists) on error
- **Write operations** return `None` or `[]` on error
- **Update/Delete operations** return `0` on error

Errors are printed to console. For production use, consider adding proper logging:

```python
import logging
from database import get_document_by_id

logging.basicConfig(level=logging.ERROR)
doc = get_document_by_id("my_db", "my_collection", "doc_id")
if doc is None:
    logging.error("Failed to fetch document")
```

---

## Method Summary

| Method | Description | Returns |
|--------|-------------|---------|
| `get_mongo_client()` | Get MongoDB client | `MongoClient` |
| `get_database()` | Get database instance | `Database` |
| `close_mongo_client()` | Close client connection | `None` |
| `get_document_by_id()` | Fetch document by ID | `dict \| None` |
| `get_documents_by_date_range()` | Fetch documents in date range | `list[dict]` |
| `get_documents_by_filter()` | Fetch documents by filter | `list[dict]` |
| `get_unique_values_by_filter()` | Get distinct field values | `list[Any]` |
| `write_in_mongo()` | Insert single document | `str \| None` |
| `insert_documents()` | Insert multiple documents | `list[str]` |
| `update_document()` | Update document by ID | `int` (modified count) |
| `delete_document()` | Delete document by ID | `int` (deleted count) |

