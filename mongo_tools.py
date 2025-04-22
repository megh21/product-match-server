from pymongo import MongoClient
import os
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from pprint import pprint
import argparse
import random

# Load environment variables
load_dotenv()

# MongoDB connection setup
MONGO_URI = os.getenv("MONGO_URI", "")
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client["productdb"]


def list_collections():
    """List all collections in the database with their stats"""
    collections = db.list_collection_names()
    print("\nAvailable collections:")
    for collection in collections:
        count = db[collection].count_documents({})
        size = db.command("collstats", collection)["size"] / 1024 / 1024  # Size in MB
        print(f"- {collection}")
        print(f"  Documents: {count}")
        print(f"  Size: {size:.2f} MB")
    return collections


def drop_collection(collection_name):
    """Drop a specific collection"""
    try:
        db[collection_name].drop()
        print(f"Successfully dropped collection: {collection_name}")
    except Exception as e:
        print(f"Error dropping collection {collection_name}: {str(e)}")


def get_collection_stats(collection_name, field_name=None):
    """Get statistics for a collection or specific field"""
    try:
        if field_name:
            # Get unique values and their counts for the field
            pipeline = [
                {"$group": {"_id": f"${field_name}", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
            stats = list(db[collection_name].aggregate(pipeline))
            print(f"\nStats for field '{field_name}' in {collection_name}:")
            for stat in stats:
                print(f"- {stat['_id']}: {stat['count']} documents")
        else:
            # Get collection stats
            stats = db.command("collstats", collection_name)
            print(f"\nCollection stats for {collection_name}:")
            pprint(stats)

    except Exception as e:
        print(f"Error getting stats: {str(e)}")


def verify_connection():
    """Verify MongoDB connection"""
    try:
        client.admin.command("ping")
        print("MongoDB connection successful!")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {str(e)}")
        return False


def add_product_costs():
    """Add random product costs to each product in the collection"""
    collections = list_collections()
    for collection in collections:
        try:
            # Get all documents in the collection
            documents = list(db[collection].find({}))
            updated_count = 0

            # Update each document with a unique random cost
            for doc in documents:
                # Generate a unique random cost between 10.0 and 100.0 for each product
                cost = round(random.uniform(10.0, 100.0), 2)
                db[collection].update_one({"_id": doc["_id"]}, {"$set": {"cost": cost}})
                updated_count += 1

            print(
                f"Added unique random costs to {updated_count} products in {collection}"
            )
        except Exception as e:
            print(f"Error adding costs to {collection}: {str(e)}")


def ensure_indexes():
    """Ensure required indexes exist"""
    collections = list_collections()
    for collection in collections:
        try:
            db[collection].create_index([("product_id", 1)], unique=True)
            print(f"Ensured index on product_id for {collection}")
        except Exception as e:
            print(f"Error ensuring index for {collection}: {str(e)}")


if __name__ == "__main__":
    if verify_connection():
        parser = argparse.ArgumentParser(description="MongoDB Utility Script")
        parser.add_argument(
            "--list", action="store_true", help="List all collections and their stats"
        )
        parser.add_argument("--drop", type=str, help="Drop a specific collection")
        parser.add_argument(
            "--stats", type=str, help="Get stats for a specific collection or field"
        )
        parser.add_argument(
            "--add-costs", action="store_true", help="Add product costs to each product"
        )
        parser.add_argument(
            "--ensure-indexes",
            action="store_true",
            help="Ensure indexes exist for all collections",
        )
        args = parser.parse_args()
        if args.list:
            list_collections()
        elif args.drop:
            drop_collection(args.drop)
        elif args.stats:
            collection_name, field_name = (
                args.stats.split(":") if ":" in args.stats else (args.stats, None)
            )
            get_collection_stats(collection_name, field_name)
        elif args.add_costs:
            add_product_costs()
        else:
            print("No valid command provided. Use --help for more information.")
