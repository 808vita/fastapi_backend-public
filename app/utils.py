import csv
from io import StringIO
import re
from fastapi import HTTPException
import google.generativeai as genai
import base64
from typing import List, Optional
import os
from dotenv import load_dotenv
import json
from google.ai.generativelanguage_v1beta.types import content
import httpx
from urllib.parse import urlencode
import asyncio
from playwright.async_api import async_playwright
from datetime import datetime

from pymongo import DESCENDING

from app.schemas import ReportedAd, SearchTerm
from .mongodb import connect_to_mongodb
from bson import ObjectId

load_dotenv()


# Gemini API Key
GOOGLE_API_KEY = os.getenv("API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY is missing in the environment file")


genai.configure(api_key=GOOGLE_API_KEY)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 5000,
    # "response_mime_type": "text/plain",
    "response_mime_type": "application/json",
}

generation_config_plain_text = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1000,
    "response_mime_type": "text/plain",
}


supported_image_mime_formats = [
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
]


async def analyze_text_with_gemini(text: str) -> dict:
    """Analyzes the given text with Gemini for scam detection."""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    prompt = f"""
        Analyze the following text for signs of a scam.
        Identify potential keywords and determine if it's a scam based on common scam patterns.
        Respond with JSON in the following format:
        {{
            "is_scam": true/false,
            "reason": "brief reason if is_scam is true else null",
            "keywords": ["keyword1", "keyword2", ...] or null
        }}
        Text: {text}
    """

    response = await model.generate_content_async(prompt)
    if response and response.text:
        try:
            json_response = json.loads(response.text)
            return json_response
        except json.JSONDecodeError:
            return {
                "gemini_response": response.text,
                "is_scam": False,
                "reason": "Could not parse gemini json response",
            }
    else:
        return {
            "gemini_response": "Error: No response from the model or response not available.",
            "is_scam": False,
        }


async def analyze_image_with_gemini(image_base64: str, mime_type: str) -> dict:
    """Analyzes the given image with Gemini for scam detection.

    base64 string might start like below sample
    data:image/jpeg;base64,/9j/4AAQS

    image parting has to start after the first "," comma
    """

    image_data = image_base64.split(",")[1]

    if mime_type not in supported_image_mime_formats:
        raise HTTPException(
            status_code=500,
            detail="valid image file needs to encoded into base64 string",
        )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    image_part = {
        "mime_type": mime_type,
        "data": image_data,
    }

    prompt = """
        Analyze the following image for signs of a scam.
        Identify potential visual cues and determine if it's a scam based on common scam patterns.
         Respond with JSON in the following format:
        {{
            "is_scam": true/false,
            "reason": "brief reason if is_scam is true else null",
            "keywords": ["keyword1", "keyword2", ...] or null
        }}
    """
    response = await model.generate_content_async([prompt, image_part])
    if response and response.text:
        try:
            json_response = json.loads(response.text)
            return json_response
        except json.JSONDecodeError:
            return {
                "gemini_response": response.text,
                "is_scam": False,
                "reason": "Could not parse gemini json response",
            }
    else:
        return {
            "gemini_response": "Error: No response from the model or response not available.",
            "is_scam": False,
        }


async def analyze_multimodal_with_gemini(
    text: str, image_base64: str, mime_type: str
) -> dict:
    """Analyzes the given text and image with Gemini for scam detection.
    base64 string might start like below sample
    data:image/jpeg;base64,/9j/4AAQS

    image parting has to start after the first "," comma
    """

    image_data = image_base64.split(",")[1]

    if mime_type not in supported_image_mime_formats:
        raise HTTPException(
            status_code=500,
            detail="valid image file needs to encoded into base64 string",
        )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    image_part = {
        "mime_type": mime_type,
        "data": image_data,
    }

    prompt = f"""
        Analyze the following text and image for signs of a scam.
        Identify potential keywords based on text and visual cues based on image and determine if it's a scam based on common scam patterns.
         Respond with JSON in the following format:
        {{
            "is_scam": true/false,
            "reason": "brief reason if is_scam is true else null",
            "keywords": ["keyword1", "keyword2", ...] or null
        }}
        Text: {text}
    """
    response = await model.generate_content_async([prompt, image_part])
    if response and response.text:
        try:
            json_response = json.loads(response.text)
            return json_response
        except json.JSONDecodeError:
            return {
                "gemini_response": response.text,
                "is_scam": False,
                "reason": "Could not parse gemini json response",
            }
    else:
        return {
            "gemini_response": "Error: No response from the model or response not available.",
            "is_scam": False,
        }


async def analyze_with_search_grounding(query: str) -> dict:
    """Analyzes the given query using google search grounding with Gemini API."""

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config_plain_text,
        tools=[
            genai.protos.Tool(
                google_search_retrieval=genai.protos.GoogleSearchRetrieval(
                    dynamic_retrieval_config=genai.protos.DynamicRetrievalConfig(
                        mode=genai.protos.DynamicRetrievalConfig.Mode.MODE_DYNAMIC,
                        dynamic_threshold=0.3,
                    ),
                ),
            ),
        ],
    )

    #   chat_session = model.start_chat(history=[])
    #   response = chat_session.send_message(query)
    response = await model.generate_content_async(query)

    if response and response.text:
        return {"gemini_response": response.text}
    else:
        return {
            "gemini_response": "Error: No response from the model or response not available."
        }


async def get_meta_ads(
    limit: int = 10,
    after: str | None = None,
    ad_delivery_date_min: str | None = None,
    ad_delivery_date_max: str | None = None,
    search_terms: str | None = None,
    ad_reached_countries: str | None = None,
    media_type: str | None = None,
    ad_active_status: str | None = None,
    search_type: str | None = None,
    ad_type: str | None = None,
    languages: str | None = None,
    publisher_platforms: str | None = None,
    search_page_ids: str | None = None,
    unmask_removed_content: str | None = None,
):
    """Fetches ads data from Meta Ad Library API."""
    client = None
    try:
        client = await connect_to_mongodb()
        if not ad_reached_countries:
            raise HTTPException(
                status_code=400, detail="ad_reached_countries parameter is required"
            )
        token_data = await fetch_access_token(client.get_database())
        access_token = token_data.get("token") if token_data else None
        if not access_token:
            raise HTTPException(status_code=500, detail="Access token not found in db.")
        base_url = "https://graph.facebook.com/v21.0/ads_archive?fields=ad_creative_link_captions,ad_creative_link_descriptions,ad_snapshot_url,page_id,page_name,publisher_platforms,ad_delivery_date"

        params = {
            "access_token": access_token,
            "limit": limit,
            "ad_reached_countries": ad_reached_countries,
        }

        if after:
            params["after"] = after
        if ad_delivery_date_min:
            params["ad_delivery_date_min"] = ad_delivery_date_min
        if ad_delivery_date_max:
            params["ad_delivery_date_max"] = ad_delivery_date_max
        if search_terms:
            params["search_terms"] = search_terms
        if media_type:
            params["media_type"] = media_type
        if ad_active_status:
            params["ad_active_status"] = ad_active_status
        if search_type:
            params["search_type"] = search_type
        if ad_type:
            params["ad_type"] = ad_type
        if languages:
            params["languages"] = languages
        if publisher_platforms:
            params["publisher_platforms"] = publisher_platforms
        if search_page_ids:
            params["search_page_ids"] = search_page_ids
        if unmask_removed_content:
            params["unmask_removed_content"] = unmask_removed_content

        url = f"{base_url}&{urlencode(params)}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                for ad in data.get("data", []):
                    ad["access_token_removed_url"] = (
                        ad.get("ad_snapshot_url").split("&access_token=")[0]
                        if ad.get("ad_snapshot_url")
                        else None
                    )
                    ad["ad_snapshot_url"] = (
                        ""  # setting this to empty as per instructions
                    )

                return data
            except httpx.HTTPError as e:
                print(f"HTTP Error:{e}")
                raise HTTPException(status_code=e.response.status_code, detail=str(e))

            except Exception as e:
                print(f"Error fetching meta ads:{e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to fetch Meta ads data:{e}"
                )
    except Exception as e:
        print(f"Error accessing or updating db:{e}")
        raise HTTPException(status_code=500, detail=f"Error during processing:{e}")


async def get_full_page_screenshot(url: str) -> str:
    """Takes a full-page screenshot of a given URL and returns it as a base64 string."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            screenshot_data = await page.screenshot(full_page=True)
            await browser.close()

            base64_string = base64.b64encode(screenshot_data).decode("utf-8")
            print(base64_string)
            return base64_string
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to take screenshot: {e}, Error : {str(e)}"
        )


async def test_mongodb_connection(db):
    """Tests the mongodb connection."""
    test_collection = db.get_collection("test_collection")
    try:
        test_data = {"message": "hello from fastapi", "created_at": datetime.now()}
        result = await test_collection.insert_one(test_data)
        inserted_id = result.inserted_id
        print("Inserted ID", inserted_id)
        retrieved_data = await test_collection.find_one({"_id": inserted_id})
        if retrieved_data:
            retrieved_data["_id"] = str(retrieved_data["_id"])
        return retrieved_data
    except Exception as e:
        print(f"Error accessing or inserting data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to access MongoDB: {e}")


async def fetch_result_limits(db):
    """Fetches result limits settings from MongoDB."""
    settings_collection = db.get_collection("settings")
    try:
        settings = await settings_collection.find_one()
        if settings:
            settings["_id"] = str(settings["_id"])
        return settings
    except Exception as e:
        print(f"Error accessing settings data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error accessing settings data:{e}"
        )


async def update_result_limits(db, results_limit: int):
    """Updates result limits in MongoDB."""
    settings_collection = db.get_collection("settings")
    try:
        update_data = {"results_limit": results_limit, "updated_at": datetime.now()}
        result = await settings_collection.find_one_and_update(
            {}, {"$set": update_data}, upsert=True, return_document=True
        )
        if result:
            result["_id"] = str(result["_id"])
        return result
    except Exception as e:
        print(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating settings:{e}")


async def create_ad_snapshot_url(base_url: str, access_token: str) -> str:
    return f"{base_url}&access_token={access_token}"


async def fetch_access_token(db):
    """Fetches access token from db"""
    access_token_collection = db.get_collection("access_token")
    try:
        token = await access_token_collection.find_one()
        if token:
            token["_id"] = str(token["_id"])
        return token
    except Exception as e:
        print(f"Error accessing access token: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing access token:{e}")


async def update_access_token(db, token: str):
    """Updates access token to db"""
    access_token_collection = db.get_collection("access_token")
    try:
        update_data = {"token": token, "updated_at": datetime.now()}
        result = await access_token_collection.find_one_and_update(
            {}, {"$set": update_data}, upsert=True, return_document=True
        )
        if result:
            result["_id"] = str(result["_id"])
        return result
    except Exception as e:
        print(f"Error updating access token: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating access token:{e}")


async def fetch_all_search_terms(db) -> List[SearchTerm]:
    """Fetches all search terms with complete data from mongodb including ids"""
    search_term_collection = db.get_collection("search_terms")
    try:
        terms = await search_term_collection.find().to_list(length=None)
        for term in terms:
            term["id"] = str(term["_id"])
            term.pop("_id")
        return terms  # type: ignore
    except Exception as e:
        print(f"Error accessing search terms data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch search terms from db:{e}"
        )


async def fetch_search_terms(db):
    """Fetches all search terms from mongodb"""
    search_term_collection = db.get_collection("search_terms")
    try:
        terms = await search_term_collection.find().to_list(length=None)
        formatted_terms = []
        for term in terms:
            formatted_terms.append(term["term"])
            if term.get("translated_terms"):
                formatted_terms.extend(term["translated_terms"])
        return formatted_terms
    except Exception as e:
        print(f"Error accessing search terms data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch search terms from db:{e}"
        )


async def create_search_term(
    db,
    term: str,
    translated_terms: Optional[List[str]] = None,
    extracted: Optional[bool] = None,
):
    """Creates a new search term to db"""
    search_term_collection = db.get_collection("search_terms")
    try:
        existing_term = await search_term_collection.find_one(
            {"term": {"$regex": f"^{re.escape(term)}$", "$options": "i"}}
        )  # case insensitive
        if existing_term:
            raise HTTPException(
                status_code=400, detail=f"Search term '{term}' already exists."
            )
        new_search_term = {
            "term": term,
            "translated_terms": translated_terms,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "extracted": extracted if extracted is not None else False,
        }
        result = await search_term_collection.insert_one(new_search_term)
        inserted_id = result.inserted_id
        retrieved_data = await search_term_collection.find_one({"_id": inserted_id})
        if retrieved_data:
            retrieved_data["_id"] = str(retrieved_data["_id"])
        return retrieved_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error creating search term: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add search term:{e}")


async def update_search_term(
    db,
    term_id: str,
    term: str,
    translated_terms: Optional[List[str]] = None,
    extracted: Optional[bool] = None,
):
    """Update a search term to db"""
    search_term_collection = db.get_collection("search_terms")
    try:
        existing_term = await search_term_collection.find_one(
            {
                "term": {"$regex": f"^{re.escape(term)}$", "$options": "i"},
                "_id": {"$ne": ObjectId(term_id)},
            }
        )  # case insensitive
        if existing_term:
            raise HTTPException(
                status_code=400, detail=f"Search term '{term}' already exists."
            )

        update_data = {
            "term": term,
            "translated_terms": translated_terms,
            "updated_at": datetime.now(),
            "extracted": extracted if extracted is not None else False,
        }
        result = await search_term_collection.find_one_and_update(
            {"_id": ObjectId(term_id)}, {"$set": update_data}, return_document=True
        )
        if result:
            result["_id"] = str(result["_id"])
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error updating search term: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update search term:{e}")


async def delete_search_term(db, term_id: str):
    """Deletes a search term to db"""
    search_term_collection = db.get_collection("search_terms")
    try:
        result = await search_term_collection.find_one_and_delete(
            {"_id": ObjectId(term_id)}
        )
        if result:
            result["_id"] = str(result["_id"])
        return result
    except Exception as e:
        print(f"Error deleting search term: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete search term:{e}")


async def fetch_search_term_by_id(db, term_id: str):
    """Fetches a search term by id from mongodb"""
    search_term_collection = db.get_collection("search_terms")
    try:
        term = await search_term_collection.find_one({"_id": ObjectId(term_id)})
        if term:
            term["_id"] = str(term["_id"])
            if term.get("translated_terms"):
                term["translated_terms"] = term["translated_terms"]
        return term
    except Exception as e:
        print(f"Error accessing search terms data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch search terms from db:{e}"
        )


async def analyze_ad_with_gemini(
    ad_screenshot_base64: Optional[str], web_screenshot_base64: Optional[str]
) -> Optional[dict]:
    """Analyzes the given ad data with Gemini for scam detection."""
    if not ad_screenshot_base64 and not web_screenshot_base64:
        return None
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    prompt = """
        Analyze the following image for signs of a scam.
        Identify potential visual cues and determine if it's a scam based on common scam patterns.
         Respond with JSON in the following format:
        {{
            "is_scam": true/false,
            "reason": "brief reason if is_scam is true else null",
            "keywords": ["keyword1", "keyword2", ...] or null
        }}
    """
    parts = []
    if ad_screenshot_base64:
        try:
            image_data = ad_screenshot_base64
            image_part = {"mime_type": "image/png", "data": image_data}
            parts.append(image_part)
        except Exception as e:
            print(f"Error decoding ad_screenshot_base64 :{e}")
    if web_screenshot_base64:
        try:
            image_data_web = web_screenshot_base64
            image_part_web = {"mime_type": "image/png", "data": image_data_web}
            parts.append(image_part_web)
        except Exception as e:
            print(f"Error decoding web_screenshot_base64 : {e}")

    if not parts:
        return None

    response = await model.generate_content_async([prompt, *parts])
    if response and response.text:
        try:
            json_response = json.loads(response.text)
            return json_response
        except json.JSONDecodeError:
            return {
                "gemini_response": response.text,
                "is_scam": False,
                "reason": "Could not parse gemini json response",
            }
    else:
        return {
            "gemini_response": "Error: No response from the model or response not available.",
            "is_scam": False,
        }


async def store_public_awareness_info(
    db,
    category_name: str,
    common_pattern: str,
    potential_user_targets: Optional[List[str]] = None,
):
    """Stores public awareness information into the category collection in MongoDB."""
    category_collection = db.get_collection("category")
    try:
        new_category_info = {
            "category_name": category_name,
            "common_pattern": common_pattern,
            "potential_user_targets": potential_user_targets,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        result = await category_collection.insert_one(new_category_info)
        inserted_id = result.inserted_id
        retrieved_data = await category_collection.find_one({"_id": inserted_id})
        if retrieved_data:
            retrieved_data["_id"] = str(retrieved_data["_id"])
        return retrieved_data
    except Exception as e:
        print(f"Error storing public awareness info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to store public awareness info:{e}"
        )


async def fetch_all_category_names(db):
    """Fetches all category names from mongodb"""
    category_collection = db.get_collection("category")
    try:
        categories = await category_collection.find().to_list(length=None)
        formatted_categories = []
        for category in categories:
            formatted_categories.append(category["category_name"])
        return formatted_categories
    except Exception as e:
        print(f"Error accessing category names data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch category names from db:{e}"
        )


async def fetch_all_categories(db):
    """Fetches all category data from mongodb"""
    category_collection = db.get_collection("category")
    try:
        categories = await category_collection.find().to_list(length=None)
        for category in categories:
            category["id"] = str(category["_id"])
            category.pop("_id")
        return categories
    except Exception as e:
        print(f"Error accessing category data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch category data from db:{e}"
        )


async def translate_text_with_gemini(text: str) -> dict:
    """Translates the given text into Tamil, Hindi, and Telugu using Gemini API."""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 500,
            # "response_mime_type": "text/plain",
            "response_mime_type": "application/json",
        },
    )

    prompt = f"""
        Translate the following text into Tamil, Hindi, and Telugu.
        Respond with JSON in the following format:
        {{
            "tamil": "tamil translation",
            "hindi": "hindi translation",
            "telugu": "telugu translation"
        }}
        Text: {text}
    """
    response = await model.generate_content_async(prompt)
    if response and response.text:
        try:
            json_response = json.loads(response.text)
            return json_response
        except json.JSONDecodeError:
            return {
                "gemini_response": response.text,
                "tamil": "Could not parse gemini json response",
                "hindi": "Could not parse gemini json response",
                "telugu": "Could not parse gemini json response",
            }
    else:
        return {
            "gemini_response": "Error: No response from the model or response not available.",
            "tamil": "No Translation",
            "hindi": "No Translation",
            "telugu": "No Translation",
        }


async def fetch_stats_from_db(db):
    """Fetches various statistics from the database."""
    meta_ads_collection = db.get_collection("meta_ads")
    search_term_collection = db.get_collection("search_terms")
    category_collection = db.get_collection("category")

    total_ads_scanned = await meta_ads_collection.count_documents({})
    is_scam_true_count = await meta_ads_collection.count_documents(
        {"gemini_analysis.is_scam": True}
    )
    search_terms_count = await search_term_collection.count_documents({})

    ads_per_search_term = await meta_ads_collection.aggregate(
        [
            {"$group": {"_id": "$search_term", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
        ]
    ).to_list(length=None)

    ads_count_per_search_term_dict = {
        item["_id"]: item["count"] for item in ads_per_search_term
    }

    ads_per_category = await meta_ads_collection.aggregate(
        [
            {"$group": {"_id": "$category_name", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
        ]
    ).to_list(length=None)

    ads_count_per_category_dict = {
        item["_id"]: item["count"] for item in ads_per_category
    }

    total_search_terms_extracted = await search_term_collection.count_documents(
        {"extracted": True}
    )

    return {
        "total_ads_scanned": total_ads_scanned,
        "is_scam_true_count": is_scam_true_count,
        "search_terms_count": search_terms_count,
        "ads_count_per_search_term": ads_count_per_search_term_dict,
        "ads_count_per_category": ads_count_per_category_dict,
        "total_search_terms_extracted": total_search_terms_extracted,
    }


async def create_reported_ad(db, ad_id: str, report_reason: str):
    """Creates a new reported ad to db"""
    reported_ads_collection = db.get_collection("reported_ads")
    try:
        new_reported_ad = {
            "ad_id": ad_id,
            "report_reason": report_reason,
            "reported": False,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        result = await reported_ads_collection.insert_one(new_reported_ad)
        inserted_id = result.inserted_id
        retrieved_data = await reported_ads_collection.find_one({"_id": inserted_id})
        if retrieved_data:
            retrieved_data["_id"] = str(retrieved_data["_id"])
        return retrieved_data
    except Exception as e:
        print(f"Error creating reported ad: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add reported ad:{e}")


async def fetch_reported_ad_by_id(db, ad_id: str):
    """Fetches a reported ad by id from mongodb"""
    reported_ads_collection = db.get_collection("reported_ads")
    try:
        reported_ad = await reported_ads_collection.find_one({"ad_id": ad_id})
        if reported_ad:
            reported_ad["_id"] = str(reported_ad["_id"])
        return reported_ad
    except Exception as e:
        print(f"Error accessing reported ad data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch reported ad from db:{e}"
        )


async def update_reported_ad(db, ad_id: str, reported: str):
    """Update an existing reported ad to db"""
    reported_ads_collection = db.get_collection("reported_ads")
    try:
        update_data = {"reported": reported, "updated_at": datetime.now()}
        result = await reported_ads_collection.find_one_and_update(
            {"ad_id": ad_id}, {"$set": update_data}, return_document=True
        )
        if result:
            result["_id"] = str(result["_id"])
        return result
    except Exception as e:
        print(f"Error updating reported ad: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update reported ad:{e}")


async def delete_reported_ad(db, ad_id: str):
    """Deletes a reported ad to db"""
    reported_ads_collection = db.get_collection("reported_ads")
    try:
        result = await reported_ads_collection.find_one_and_delete({"ad_id": ad_id})
        if result:
            result["_id"] = str(result["_id"])
        return result
    except Exception as e:
        print(f"Error deleting reported ad: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete reported ad:{e}")


async def fetch_all_reported_ads(db) -> List[ReportedAd]:
    """Fetches all reported ads from mongodb"""
    reported_ads_collection = db.get_collection("reported_ads")
    try:
        reported_ads = await reported_ads_collection.find().to_list(length=None)
        for reported_ad in reported_ads:
            reported_ad["id"] = str(reported_ad["_id"])
            reported_ad.pop("_id")
        return reported_ads  # type: ignore
    except Exception as e:
        print(f"Error accessing reported ads data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch reported ads from db:{e}"
        )


async def fetch_pending_reported_ads(db) -> List[ReportedAd]:
    """Fetches all reported ads with 'reported' set to false from mongodb."""
    reported_ads_collection = db.get_collection("reported_ads")
    try:
        reported_ads = await reported_ads_collection.find({"reported": False}).to_list(
            length=None
        )
        for reported_ad in reported_ads:
            reported_ad["id"] = str(reported_ad["_id"])
            reported_ad.pop("_id")
        return reported_ads
    except Exception as e:
        print(f"Error accessing pending reported ads data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch pending reported ads from db:{e}"
        )


def generate_csv_from_data(data: List[dict]) -> str:
    """Converts a list of dictionaries to CSV format."""
    if not data:
        return ""

    output = StringIO()
    if data:
        csv_writer = csv.DictWriter(output, fieldnames=data[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(data)
    return output.getvalue()
