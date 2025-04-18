from pymongo import MongoClient
import os
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from pprint import pprint
import argparse
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
                {"$sort": {"count": -1}}
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

if __name__ == "__main__":
    if verify_connection():
        parser = argparse.ArgumentParser(description="MongoDB Utility Script")
        parser.add_argument("--list", action="store_true", help="List all collections and their stats")
        parser.add_argument("--drop", type=str, help="Drop a specific collection")
        parser.add_argument("--stats", type=str, help="Get stats for a specific collection or field")
        args = parser.parse_args()
        if args.list:
            list_collections()
        elif args.drop:
            drop_collection(args.drop)
        elif args.stats:
            collection_name, field_name = args.stats.split(":") if ":" in args.stats else (args.stats, None)
            get_collection_stats(collection_name, field_name)
        else:
            print("No valid command provided. Use --help for more information.")