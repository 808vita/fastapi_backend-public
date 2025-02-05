from fastapi import APIRouter, HTTPException
from ..schemas import (
    TextAnalysisRequest,
    ImageAnalysisRequest,
    MultimodalAnalysisRequest,
    SearchGroundingRequest,
    TranslationRequest,
)
from ..utils import (
    analyze_text_with_gemini,
    analyze_image_with_gemini,
    analyze_multimodal_with_gemini,
    analyze_with_search_grounding,
    translate_text_with_gemini,
)

router = APIRouter()


@router.post("/analyze/text", response_model=dict)
async def analyze_text(request: TextAnalysisRequest):
    """Analyzes text for scam patterns using Gemini API"""
    try:
        result = await analyze_text_with_gemini(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/image", response_model=dict)
async def analyze_image(request: ImageAnalysisRequest):
    """Analyzes image for scam patterns using Gemini API"""
    try:
        result = await analyze_image_with_gemini(
            request.image_base64, request.mime_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/multimodal", response_model=dict)
async def analyze_multimodal(request: MultimodalAnalysisRequest):
    """Analyzes both text and image for scam patterns using Gemini API"""
    try:
        result = await analyze_multimodal_with_gemini(
            request.text, request.image_base64, request.mime_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/search", response_model=dict)
async def analyze_search(request: SearchGroundingRequest):
    """Analyzes query using search grounding with Gemini API"""
    try:
        result = await analyze_with_search_grounding(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New Translation API endpoint
@router.post("/translate", response_model=dict)
async def translate(request: TranslationRequest):
    """Translates text into Tamil, Hindi, and Telugu using Gemini API"""
    try:
        result = await translate_text_with_gemini(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
