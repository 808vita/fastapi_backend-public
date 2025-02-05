import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()
# Retrieve database URI from environment variables
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing in the .env file")


# Create a reusable database connection function.
async def connect_to_mongodb():
    try:
        client = AsyncIOMotorClient(MONGODB_URI, server_api=ServerApi("1"))
        await client.admin.command("ping")  # verify if the db connection is working
        return client
    except Exception as e:
        print(f"Error connecting to mongodb {e}")
        raise HTTPException(status_code=500, detail=f"Error connecting to MongoDB: {e}")


async def close_mongodb_connection(client):
    try:
        if client:
            client.close()
    except Exception as e:
        print(f"Error closing mongodb connection:{e}")
        raise HTTPException(
            status_code=500, detail=f"Error while closing mongodb connection:{e}"
        )
