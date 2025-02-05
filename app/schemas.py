from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class TextAnalysisRequest(BaseModel):
    text: str


class ImageAnalysisRequest(BaseModel):
    image_base64: str
    mime_type: str


class MultimodalAnalysisRequest(BaseModel):
    text: str
    image_base64: str
    mime_type: str


class SearchGroundingRequest(BaseModel):
    query: str


class AnalysisResponse(BaseModel):
    is_scam: bool
    reason: Optional[str] = None
    keywords: Optional[List[str]] = None
    gemini_response: Optional[str] = None


class MetaAdsRequest(BaseModel):
    limit: Optional[int] = 10
    after: Optional[str] = None
    ad_delivery_date_min: Optional[str] = None
    ad_delivery_date_max: Optional[str] = None
    search_terms: Optional[str] = None
    ad_reached_countries: str
    media_type: Optional[str] = None
    ad_active_status: Optional[str] = None
    search_type: Optional[str] = None
    ad_type: Optional[str] = None
    languages: Optional[str] = None
    publisher_platforms: Optional[str] = None
    search_page_ids: Optional[str] = None
    unmask_removed_content: Optional[str] = None


class ScreenshotRequest(BaseModel):
    url: str


class SearchTerm(BaseModel):
    id: Optional[str] = None  # Add id field
    term: str
    translated_terms: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    extracted: bool = False


class Settings(BaseModel):
    results_limit: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AccessToken(BaseModel):
    token: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MetaAd(BaseModel):
    ad_creative_link_captions: Optional[List[str]] = None
    ad_creative_link_descriptions: Optional[List[str]] = None
    ad_snapshot_url: str
    page_id: str
    page_name: str
    publisher_platforms: List[str]
    ad_delivery_date: str
    screenshot_base64: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    access_token_removed_url: Optional[str] = None


class CheckedMetaAd(BaseModel):
    ad_creative_link_captions: Optional[List[str]] = None
    ad_creative_link_descriptions: Optional[List[str]] = None
    ad_snapshot_url: str
    page_id: str
    page_name: str
    publisher_platforms: List[str]
    ad_delivery_date: Optional[str] = None
    ad_screenshot_base64: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    access_token_removed_url: Optional[str] = None
    search_term: Optional[str] = None
    web_screenshot_base64: Optional[str] = None
    id: str
    gemini_analysis: Optional[dict] = None
    category_name: Optional[str] = None
    potential_user_targets: Optional[List[str]] = None


class ReportedAd(BaseModel):
    ad_id: str
    report_reason: str
    reported_at: Optional[datetime] = None


class UserProfile(BaseModel):
    user_id: str
    preferences: Optional[dict[str, str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TestData(BaseModel):
    message: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PublicAwarenessRequest(BaseModel):
    category_name: str
    common_pattern: str
    potential_user_targets: Optional[List[str]] = None


class Category(BaseModel):
    category_name: str
    common_pattern: str
    potential_user_targets: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# New Translation Request Model
class TranslationRequest(BaseModel):
    text: str


class ReportedAd(BaseModel):
    ad_id: str
    report_reason: Optional[str] = None
    reported: bool = False  # Add a 'reported' flag
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
