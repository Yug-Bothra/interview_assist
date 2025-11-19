"""
AI Models Routes
Model availability and configuration
"""

from fastapi import APIRouter

from services.ai_service import get_available_models, get_response_styles
from config.settings import settings

router = APIRouter()


@router.get("/models/status")
async def get_model_status():
    """
    Get available AI models and current default
    """
    return {
        "default_provider": settings.DEFAULT_MODEL,
        "available_providers": get_available_models()
    }


@router.get("/models/list")
async def list_models():
    """
    List all available AI models with details
    """
    models = get_available_models()
    return {
        "models": [
            {
                "id": model_id,
                "name": model_id,
                "available": available,
                "is_default": model_id == settings.DEFAULT_MODEL
            }
            for model_id, available in models.items()
        ]
    }


@router.get("/response-styles")
async def get_response_style_list():
    """
    Get available response styles for interview answers
    """
    styles = get_response_styles()
    return {
        "styles": [
            {
                "id": style_id,
                "name": config["name"],
                "description": config["prompt"][:100] + "..."
            }
            for style_id, config in styles.items()
        ]
    }