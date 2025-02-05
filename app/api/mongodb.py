from bson import ObjectId
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..schemas import (
    Category,
    CheckedMetaAd,
    MetaAd,
    PublicAwarenessRequest,
    ReportedAd,
    SearchTerm,
    Settings,
    TestData,
    AccessToken,
)
from ..utils import (
    create_reported_ad,
    delete_reported_ad,
    fetch_all_categories,
    fetch_all_reported_ads,
    fetch_all_search_terms,
    fetch_pending_reported_ads,
    fetch_reported_ad_by_id,
    fetch_search_term_by_id,
    fetch_stats_from_db,
    store_public_awareness_info,
    test_mongodb_connection,
    fetch_result_limits,
    update_reported_ad,
    update_result_limits,
    fetch_access_token,
    update_access_token,
    fetch_search_terms,
    create_search_term,
    update_search_term,
    delete_search_term,
    generate_csv_from_data,
)
from ..mongodb import connect_to_mongodb
from typing import List, Optional
from pymongo import DESCENDING


router = APIRouter()


@router.get("/test-db", response_model=dict)
async def test_db():
    """Tests the mongodb connection."""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await test_mongodb_connection(db)
        return {"mongodb_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings", response_model=dict)
async def get_settings():
    """Get result limits from mongodb"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_result_limits(db)
        return {"settings": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/settings", response_model=dict)
async def update_settings(request: Settings):
    """Updates result limit settings to mongodb"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await update_result_limits(db, request.results_limit)
        return {"settings": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/access-token", response_model=dict)
async def get_access_token():
    """Gets the access token from db"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_access_token(db)
        return {"access_token": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/access-token", response_model=dict)
async def update_access_token_api(request: AccessToken):
    """Updates the access token in db"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await update_access_token(db, request.token)
        return {"access_token": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/access-token", response_model=dict)
async def update_access_token_api(request: AccessToken):
    """Updates the access token in db"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await update_access_token(db, request.token)
        return {"access_token": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-terms", response_model=List[SearchTerm])
async def get_all_search_terms_api():
    """Gets all search terms with complete data including ids"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_all_search_terms(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-terms/{term_id}", response_model=dict)
async def get_search_term_api(term_id: str):
    """Gets search term by id"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_search_term_by_id(db, term_id)
        return {"search_term": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-terms", response_model=dict)
async def create_search_term_api(request: SearchTerm):
    """Create a new search term"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await create_search_term(db, request.term, request.translated_terms)
        return {"search_term": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/search-terms/{term_id}", response_model=dict)
async def update_search_term_api(term_id: str, request: SearchTerm):
    """Update a search term"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await update_search_term(
            db, term_id, request.term, request.translated_terms
        )
        return {"search_term": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/search-terms/{term_id}", response_model=dict)
async def delete_search_term_api(term_id: str):
    """Delete a search term"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await delete_search_term(db, term_id)
        return {"search_term": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-terms-all", response_model=List[str])
async def get_search_terms_all():
    """Gets all search terms including translated versions"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_search_terms(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checked-meta-ads", response_model=List[CheckedMetaAd])
async def get_checked_meta_ads_api(limit: int = 10, offset: int = 0):
    """Gets all ads from db with offset-based pagination, latest first."""
    if limit > 10:
        limit = 10  # limit the max value

    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        meta_ads_collection = db.get_collection("meta_ads")

        # Fetch ads with sorting and pagination
        ads = (
            await meta_ads_collection.find()
            .sort("created_at", DESCENDING)  # Sort by created_at descending
            .skip(offset)  # Apply the offset
            .limit(limit)  # Apply the limit
            .to_list(length=limit)
        )

        # Convert ObjectIds to strings for serialization
        for ad in ads:
            ad["_id"] = str(ad["_id"])
        return ads
    except Exception as e:
        print(f"Error accessing meta ads data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch meta ads from db: {e}"
        )


@router.get("/checked-meta-ads/{ad_id}", response_model=dict)
async def get_checked_meta_ad_by_id_api(ad_id: str):
    """Gets a single ad from db by its ID."""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        meta_ads_collection = db.get_collection("meta_ads")
        ad = await meta_ads_collection.find_one({"id": ad_id})
        if ad:
            ad["_id"] = str(ad["_id"])
        return {"ad": ad}
    except Exception as e:
        print(f"Error accessing meta ads data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch meta ad from db: {e}"
        )


@router.get("/checked-meta-ads-unsorted", response_model=List[CheckedMetaAd])
async def get_checked_meta_ads_api(limit: int = 10, offset: int = 0):
    """Gets all ads from db with offset-based pagination, latest first."""
    if limit > 10:
        limit = 10  # limit the max value

    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        meta_ads_collection = db.get_collection("meta_ads")

        # Fetch ads with sorting and pagination
        ads = (
            await meta_ads_collection.find()
            # .sort("created_at", DESCENDING)  # Sort by created_at descending 
            .skip(offset)  # Apply the offset
            .limit(limit)  # Apply the limit
            .to_list(length=limit)
        )

        # Convert ObjectIds to strings for serialization
        for ad in ads:
            ad["_id"] = str(ad["_id"])
        return ads
    except Exception as e:
        print(f"Error accessing meta ads data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch meta ads from db: {e}"
        )


@router.post("/public-awareness", response_model=dict)
async def create_public_awareness_api(request: PublicAwarenessRequest):
    """Creates public awareness data"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await store_public_awareness_info(
            db,
            request.category_name,
            request.common_pattern,
            request.potential_user_targets,
        )
        return {"public_awareness_data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories", response_model=List[Category])
async def get_all_categories_api():
    """Gets all category data from db"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_all_categories(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=dict)  # Update response model
async def get_stats_api():
    """Gets stats data"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_stats_from_db(db)
        return {"stats": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reported-ads/", response_model=dict)
async def create_reported_ad_api(request: ReportedAd):
    """Create a new reported ad"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await create_reported_ad(db, request.ad_id, request.report_reason)
        return {"reported_ad": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reported-ads/{ad_id}", response_model=dict)
async def get_reported_ad_api(ad_id: str):
    """Get a reported ad by ad_id"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_reported_ad_by_id(db, ad_id)
        return {"reported_ad": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/reported-ads/{ad_id}", response_model=dict)
async def update_reported_ad_api(ad_id: str, request: ReportedAd):
    """Update an existing reported ad"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await update_reported_ad(db, ad_id, request.reported)
        return {"reported_ad": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reported-ads/{ad_id}", response_model=dict)
async def delete_reported_ad_api(ad_id: str):
    """Delete a reported ad"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await delete_reported_ad(db, ad_id)
        return {"reported_ad": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reported-ads-all", response_model=List[ReportedAd])
async def get_all_reported_ads_api():
    """Gets all reported ads"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_all_reported_ads(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reported-ads-pending", response_model=List[ReportedAd])
async def get_pending_reported_ads_api():
    """Gets all reported ads with 'reported' set to false."""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        result = await fetch_pending_reported_ads(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/reported-ads")
async def export_reported_ads_api():
    """Exports reported ads to CSV"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        reported_ads = await fetch_all_reported_ads(db)
        csv_data = generate_csv_from_data(reported_ads)
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment;filename=reported_ads.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/meta-ads")
async def export_meta_ads_api():
    """Exports meta ads to CSV, excluding base64 fields"""
    try:
        client = await connect_to_mongodb()
        db = client.get_database()
        meta_ads_collection = db.get_collection("meta_ads")
        ads = await meta_ads_collection.find().to_list(length=None)
        # Remove base64 fields
        for ad in ads:
            ad.pop("web_screenshot_base64", None)
            ad.pop("ad_screenshot_base64", None)
            ad["_id"] = str(ad["_id"])
        csv_data = generate_csv_from_data(ads)
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment;filename=meta_ads.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
