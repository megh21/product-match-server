from pymongo import MongoClient, UpdateOne
import os
from pymongo.server_api import ServerApi
from pymongo.errors import BulkWriteError

# Replace with your actual Mongo URI
MONGO_URI = os.getenv("MONGO_URI", "")
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))

db = client["productdb"]
products_col = db["products"]
logs_col = db["logs"]


def get_product_by_id(product_id):
    product = products_col.find_one({"product_id": product_id})
    if product:
        # Convert ObjectId to string
        product['_id'] = str(product['_id'])
    return product


def insert_sample_products(products):
    """Insert or update products in MongoDB efficiently"""
    if not products:
        print("No products to insert")
        return
    
    # Extract product IDs from new data
    new_product_ids = [p["product_id"] for p in products]
    
    # Find existing product IDs in MongoDB
    existing_product_ids = set(
        doc["product_id"] 
        for doc in products_col.find(
            {"product_id": {"$in": new_product_ids}}, 
            {"product_id": 1}
        )
    )
    
    # Prepare operations
    operations = []
    for product in products:
        operations.append(
            UpdateOne(
                {"product_id": product["product_id"]},
                {"$set": product},
                upsert=True
            )
        )
    
    if operations:
        try:
            result = products_col.bulk_write(operations, ordered=False)
            print(f"Inserted {result.upserted_count} new products")
            print(f"Modified {result.modified_count} existing products")
        except BulkWriteError as bwe:
            print(f"Error during bulk write: {bwe.details}")
    else:
        print("No operations to perform")


def get_db_stats():
    """Get database statistics"""
    return {
        "product_count": products_col.count_documents({}),
        "last_updated": products_col.find_one(
            {}, 
            sort=[("_id", -1)]
        )["_id"].generation_time if products_col.count_documents({}) > 0 else None
    }


def ensure_indexes():
    """Ensure required indexes exist"""
    products_col.create_index([("product_id", 1)], unique=True)


if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
