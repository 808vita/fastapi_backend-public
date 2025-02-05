import os
import re
import sys
from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.scheduler_utils import fetch_ads_data_and_process
from .schemas import (
    TextAnalysisRequest,
    ImageAnalysisRequest,
    MultimodalAnalysisRequest,
    AnalysisResponse,
    SearchGroundingRequest,
    MetaAdsRequest,
    ScreenshotRequest,
    SearchTerm,
    Settings,
    AccessToken,
    MetaAd,
    ReportedAd,
    UserProfile,
    TestData,
)
from .utils import (
    analyze_ad_with_gemini,
    analyze_text_with_gemini,
    analyze_image_with_gemini,
    analyze_multimodal_with_gemini,
    analyze_with_search_grounding,
    create_ad_snapshot_url,
    fetch_access_token,
    fetch_result_limits,
    fetch_search_terms,
    get_meta_ads,
    get_full_page_screenshot,
)


import google.generativeai as genai
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .mongodb import connect_to_mongodb, close_mongodb_connection
from .api import gemini, meta, mongodb

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2000,
    # "response_mime_type": "text/plain",
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are a cat. And you like memes.",
)


# Define the scheduled task function
async def fetch_meta_ads_scheduled():
    client = None
    try:
        client = await connect_to_mongodb()
        db = client.get_database()

        settings = await fetch_result_limits(db)
        limit = settings.get("results_limit") if settings else 10
        search_terms_list = await fetch_search_terms(db)
        if not search_terms_list:
            search_terms_list = [
                "loan",
            ]
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        ad_reached_countries = "['IN']"
        token_data = await fetch_access_token(db)
        access_token = token_data.get("token") if token_data else None
        for search_term in search_terms_list:
            await fetch_ads_data_and_process(
                db, search_term, limit, yesterday, ad_reached_countries, access_token
            )
    except Exception as e:
        logging.error(f"Error Connecting to db or while doing db ops {e}")
    finally:
        if client:
            await close_mongodb_connection(client)


# Create a scheduler
scheduler = BackgroundScheduler()


def run_async_job(job_function):
    """Run an async function in an event loop using ThreadPoolExecutor."""
    asyncio.run(job_function())


# Add the scheduled job
scheduler.add_job(run_async_job, "interval", hours=2, args=[fetch_meta_ads_scheduled])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    client = None
    try:
        client = await connect_to_mongodb()
        logging.info("Mongodb connection started")
        scheduler.start()
        logging.info("Scheduler started")
        yield
    finally:
        if client:
            await close_mongodb_connection(client)
            logging.info("Mongodb connection closed")
    logging.info("Scheduler shutting down")
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:3000", "https://-production-domain.oof"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gemini.router, prefix="/api")
app.include_router(meta.router, prefix="/api")
app.include_router(mongodb.router, prefix="/api")


@app.get("/")
def read_root():
    response = model.generate_content("Explain how AI works")
    return {"hello": "world", "gemini": response.text}


@app.get("/items/")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
