"""
Configuration Settings
Centralized environment variable management
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
import openai


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Keys
    DEEPGRAM_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    SUPABASE_URL: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
    
    # Server Configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # AI Configuration
    DEFAULT_MODEL: str = "gpt-4o-mini"
    
    # WebSocket Configuration
    KEEPALIVE_INTERVAL: int = 5
    RENDER_KEEPALIVE: int = 30
    
    # Deepgram Configuration
    DEEPGRAM_MODEL: str = "nova-2"
    DEEPGRAM_LANGUAGE: str = "en"
    
    # Transcript Configuration
    PAUSE_THRESHOLD: float = 2.0
    MIN_QUESTION_LENGTH: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Initialize settings
settings = Settings()

# Configure OpenAI if key is available
if settings.OPENAI_API_KEY:
    openai.api_key = settings.OPENAI_API_KEY


def print_startup_info():
    """Print startup configuration information"""
    print("\n" + "=" * 70)
    print("⚙️  CONFIGURATION STATUS")
    print("=" * 70)
    
    print(f"✅ DEEPGRAM_API_KEY: {'Configured' if settings.DEEPGRAM_API_KEY else '❌ Missing'}")
    print(f"✅ OPENAI_API_KEY: {'Configured' if settings.OPENAI_API_KEY else '❌ Missing'}")
    print(f"✅ SUPABASE_URL: {'Configured' if settings.SUPABASE_URL else '⚠️  Optional'}")
    print(f"✅ SUPABASE_KEY: {'Configured' if settings.SUPABASE_SERVICE_ROLE_KEY else '⚠️  Optional'}")
    
    print(f"\n🔧 Server Config:")
    print(f"   - Port: {settings.PORT}")
    print(f"   - Default Model: {settings.DEFAULT_MODEL}")
    print(f"   - Keepalive: {settings.KEEPALIVE_INTERVAL}s")
    
    print("=" * 70 + "\n")


def validate_api_keys(deepgram: bool = False, openai_key: bool = False) -> tuple[bool, str]:
    """
    Validate required API keys
    
    Args:
        deepgram: Check if Deepgram key is required
        openai_key: Check if OpenAI key is required
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if deepgram and not settings.DEEPGRAM_API_KEY:
        return False, "DEEPGRAM_API_KEY not configured"
    
    if openai_key and not settings.OPENAI_API_KEY:
        return False, "OPENAI_API_KEY not configured"
    
    return True, "All required keys configured"