"""
Health Check Routes
Status and diagnostics endpoints
"""

from fastapi import APIRouter

from config.settings import settings

router = APIRouter()


@router.get("/")
async def root():
    """
    Root endpoint - Basic service info
    """
    return {
        "status": "running",
        "service": "Interview Assistant API",
        "version": "2.0.0",
        "architecture": "modular",
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health")
async def health_check():
    """
    Detailed health check with service status
    """
    return {
        "status": "healthy",
        "message": "Interview Assistant API - Modular Architecture",
        "services": {
            "deepgram": "configured" if settings.DEEPGRAM_API_KEY else "missing",
            "openai": "configured" if settings.OPENAI_API_KEY else "missing",
            "supabase": "configured" if settings.SUPABASE_URL else "optional"
        },
        "config": {
            "default_model": settings.DEFAULT_MODEL,
            "keepalive_interval": settings.KEEPALIVE_INTERVAL,
            "audio_capture": "browser-based"
        }
    }


@router.get("/api/status")
async def api_status():
    """
    API configuration status
    """
    return {
        "deepgram": {
            "configured": bool(settings.DEEPGRAM_API_KEY),
            "model": settings.DEEPGRAM_MODEL,
            "language": settings.DEEPGRAM_LANGUAGE
        },
        "openai": {
            "configured": bool(settings.OPENAI_API_KEY),
            "default_model": settings.DEFAULT_MODEL
        },
        "websockets": {
            "keepalive_interval": settings.KEEPALIVE_INTERVAL,
            "render_keepalive": settings.RENDER_KEEPALIVE
        }
    }