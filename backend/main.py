"""
Main FastAPI Application - Interview Assistant Backend
Modular, Vercel-Compatible Architecture
"""

import sys
import os

# Ensure UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

print("=" * 70)
print("🚀 STARTING INTERVIEW ASSISTANT BACKEND")
print("=" * 70)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import configuration
from config.settings import settings, print_startup_info

# Import routers
from api.routes import health, models
from api.websockets import deepgram_ws, interview_ws

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Interview Assistant API",
    description="Modular Interview Assistant with Deepgram & OpenAI",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

# Health check routes
app.include_router(health.router, tags=["Health"])

# Model status routes
app.include_router(models.router, prefix="/api", tags=["Models"])

# WebSocket routes
app.include_router(deepgram_ws.router, tags=["Deepgram"])
app.include_router(interview_ws.router, tags=["Interview"])

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Print configuration on startup"""
    print_startup_info()

# ============================================================================
# MAIN ENTRY POINT (for local development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = settings.PORT
    
    print("\n" + "=" * 70)
    print("🚀 STARTING SERVER")
    print("=" * 70)
    print(f"Port: {port}")
    print(f"Host: 0.0.0.0")
    print("=" * 70)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            timeout_keep_alive=75
        )
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)