from datetime import datetime
import os
import google.generativeai as genai
from typing import Optional
from fastapi import HTTPException
import logging
from .mongodb import connect_to_mongodb, close_mongodb_connection
from .utils import (
    get_meta_ads,
    create_ad_snapshot_url,
    get_full_page_screenshot,
    analyze_ad_with_gemini,
    fetch_access_token,
    create_search_term,
    store_public_awareness_info,  # Import to add new categories
    fetch_all_category_names,  # Import to fetch existing categories
)
import re
from bson import ObjectId
import json


# Gemini API Key
GOOGLE_API_KEY = os.getenv("API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY is missing in the environment file")


genai.configure(api_key=GOOGLE_API_KEY)


async def process_ad_and_store(db, ad, search_term, access_token, meta_ads_collection):
    existing_ad = await meta_ads_collection.find_one({"id": ad.get("id")})
    if not existing_ad:
        ad["created_at"] = datetime.now()
        ad["updated_at"] = datetime.now()
        ad["search_term"] = search_term
        full_snapshot_url = await create_ad_snapshot_url(
            ad.get("access_token_removed_url"),
            access_token,
        )
        try:
            base64_string = await get_full_page_screenshot(full_snapshot_url)
            ad["ad_screenshot_base64"] = base64_string
            logging.info(
                f"Screenshot captured for : { ad.get("access_token_removed_url")} "
            )
        except Exception as e:
            logging.error(
                f"Error while getting screenshot :{e}, url: {full_snapshot_url}"
            )
        gemini_response = None
        if ad.get("ad_creative_link_captions") and isinstance(
            ad.get("ad_creative_link_captions"), list
        ):
            for caption in ad.get("ad_creative_link_captions"):
                try:
                    base64_string = await get_full_page_screenshot("https://" + caption)
                    ad["web_screenshot_base64"] = base64_string
                    logging.info(f"Screenshot captured for : {caption}")
                    gemini_response = await analyze_ad_with_gemini(
                        ad.get("ad_screenshot_base64"),
                        ad.get("web_screenshot_base64"),
                    )
                    break  # take screenshot of first valid url
                except Exception as e:
                    logging.error(
                        f"Error while getting screenshot : {e}  , url : {caption}"
                    )
        if gemini_response:
            ad["gemini_analysis"] = gemini_response
            if (
                gemini_response
                and gemini_response.get("keywords")
                and isinstance(gemini_response.get("keywords"), list)
            ):
                for keyword in gemini_response.get("keywords"):
                    try:
                        await create_search_term(db, keyword, extracted=True)
                        logging.info(f"New keyword '{keyword}' added to db")
                    except HTTPException as e:
                        logging.info(f"Error creating term : {keyword} , error : {e}")
                    except Exception as e:
                        logging.info(f"Error creating term : {keyword} , error : {e}")

            # Start Category Analysis Logic
            if gemini_response.get("is_scam") == True:
                category_names = await fetch_all_category_names(db)
                category_data, category_match, matched_category_name = (
                    await analyze_ad_for_category(ad, gemini_response, category_names)
                )  # Updated call to include category_names

                if category_match:
                    ad["category_name"] = matched_category_name

                    logging.info(f"Category matched : {matched_category_name}")
                elif category_data:  # new category
                    created_category = await store_public_awareness_info(
                        db,
                        category_data.get("category_name"),
                        category_data.get("common_pattern"),
                        category_data.get("potential_user_targets"),
                    )
                    if created_category:
                        ad["category_name"] = created_category.get("category_name")
                        logging.info(
                            f"New category created : {created_category.get('category_name')}"
                        )
                    else:
                        logging.info("New Category creation failed")
                # Add potential_user_targets to the ad object
                if category_data and category_data.get("potential_user_targets"):
                    ad["potential_user_targets"] = category_data.get(
                        "potential_user_targets"
                    )
                # start report ad logic

                reported_ads_collection = db.get_collection("reported_ads")
                new_reported_ad = {
                    "ad_id": ad.get("id"),
                    "reported": False,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
                await reported_ads_collection.insert_one(new_reported_ad)
            # End Category Analysis Logic

        await meta_ads_collection.insert_one(ad)
        logging.info(f"Inserted ad id : {ad.get('id')}")
    else:
        logging.info(f"Ad already present id : {ad.get('id')}")


async def analyze_ad_for_category(ad, gemini_response, category_names):
    """Analyzes an ad using Gemini to extract category data and match with existing categories."""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 5000,
            "response_mime_type": "application/json",
        },
    )

    prompt = f"""
        Analyze the following ad details and gemini analysis to determine what category does it belong to based on common scam patterns.
        Also determine if this falls under any of the following existing categories.
        Existing categories: {category_names}
        Gemini Analysis: {json.dumps(gemini_response)}

        Provide category information in JSON format based on the below schema:
        {{
            "category_name": "name of the category if it is a new one , this cannot be null",
            "common_pattern": "brief description of pattern if new category",
            "potential_user_targets": ["list", "of", "user","segments", "who" ,"are" ,"targeted"] or null if new category,
            "category_match": true/false if it matches any of the given categories,
            "matched_category_name":"name of the matched category" if category_match is true else null
        }}
    """
    response = await model.generate_content_async(prompt)
    if response and response.text:
        try:
            json_response = json.loads(response.text)
            category_match = json_response.get("category_match", False)
            matched_category_name = json_response.get("matched_category_name", None)
            if not category_match:
                return json_response, False, None
            return (
                json_response,
                category_match,
                matched_category_name,
            )  # return category_match along with category_data
        except json.JSONDecodeError:
            logging.info(f"Could not parse the gemini response json {response.text}")
            return None, False, None
    else:
        logging.info(f"Error : no response or text for category extraction: {response}")
        return None, False, None


async def fetch_ads_data_and_process(
    db, search_term, limit, yesterday, ad_reached_countries, access_token
):
    try:
        ads_data = await get_meta_ads(
            limit=limit,
            search_terms=search_term,
            ad_reached_countries=ad_reached_countries,
            ad_delivery_date_min=yesterday,
        )
        logging.info(f"Successfully fetched ads for '{search_term}'")
        if ads_data and ads_data.get("data"):
            meta_ads_collection = db.get_collection("meta_ads")
            for ad in ads_data.get("data"):
                await process_ad_and_store(
                    db, ad, search_term, access_token, meta_ads_collection
                )
    except Exception as e:
        logging.error(f"Error fetching ads for '{search_term}':  , error : {str(e)}")
