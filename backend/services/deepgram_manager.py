"""
Deepgram Service Manager
Handles dual-stream transcription with Deepgram API
"""

import asyncio
import json
import time
from typing import Optional
from enum import Enum

import websockets
from websockets.exceptions import ConnectionClosed

from config.settings import settings


class StreamType(Enum):
    """Type of audio stream"""
    CANDIDATE = "candidate"
    INTERVIEWER = "interviewer"


class ConnectionState(Enum):
    """WebSocket connection state"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"


def get_deepgram_url(language: str = "en") -> str:
    """
    Generate Deepgram WebSocket URL with configuration
    
    Args:
        language: Language code (e.g., 'en', 'es', 'fr')
        
    Returns:
        Configured WebSocket URL
    """
    return (
        f"wss://api.deepgram.com/v1/listen"
        f"?model={settings.DEEPGRAM_MODEL}"
        f"&language={language}"
        f"&encoding=linear16"
        f"&sample_rate=16000"
        f"&channels=1"
        f"&interim_results=true"
        f"&punctuate=true"
        f"&smart_format=true"
        f"&endpointing=300"
        f"&utterance_end_ms=1000"
        f"&filler_words=false"
        f"&profanity_filter=false"
    )


class DeepgramStream:
    """
    Manages a single Deepgram WebSocket connection
    Handles audio streaming and transcript reception
    """
    
    def __init__(self, api_key: str, stream_type: StreamType, language: str = "en"):
        self.api_key = api_key
        self.stream_type = stream_type
        self.language = language
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.keepalive_task: Optional[asyncio.Task] = None
        self.is_closing = False
        self.state = ConnectionState.DISCONNECTED
        self.max_retries = 3
        
    async def connect(self) -> None:
        """
        Establish connection to Deepgram API
        Includes retry logic with exponential backoff
        """
        self.state = ConnectionState.CONNECTING
        
        for attempt in range(self.max_retries):
            try:
                url = get_deepgram_url(self.language)
                
                # ✅ CRITICAL: Use 'additional_headers' for websockets 14.1+
                self.ws = await websockets.connect(
                    url,
                    additional_headers={"Authorization": f"Token {self.api_key}"},
                    ping_interval=20,
                    ping_timeout=30,
                    max_size=10_000_000,
                    close_timeout=5
                )
                
                self.state = ConnectionState.CONNECTED
                emoji = "🎤" if self.stream_type == StreamType.CANDIDATE else "💻"
                print(f"{emoji} Deepgram connected ({self.stream_type.value})")
                return
                
            except Exception as e:
                print(f"❌ Deepgram connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    self.state = ConnectionState.DISCONNECTED
                    raise
    
    async def send_keepalive(self) -> None:
        """Send periodic keepalive messages to maintain connection"""
        try:
            while not self.is_closing and self.ws and self.state == ConnectionState.CONNECTED:
                await asyncio.sleep(settings.KEEPALIVE_INTERVAL)
                if self.ws and not self.is_closing:
                    try:
                        await self.ws.send(json.dumps({"type": "KeepAlive"}))
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """
        Send audio data to Deepgram
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.ws or self.is_closing or self.state != ConnectionState.CONNECTED:
            return False
        try:
            await self.ws.send(audio_data)
            return True
        except Exception:
            return False
    
    async def receive_transcripts(self) -> Optional[dict]:
        """
        Receive transcript data from Deepgram
        
        Returns:
            Parsed JSON response or None
        """
        if not self.ws or self.state != ConnectionState.CONNECTED:
            return None
        try:
            message = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
            return json.loads(message)
        except (asyncio.TimeoutError, ConnectionClosed):
            return None
        except Exception:
            return None
    
    async def close(self) -> None:
        """Gracefully close the Deepgram connection"""
        if self.is_closing:
            return
        
        self.is_closing = True
        self.state = ConnectionState.DISCONNECTING
        
        # Cancel keepalive task
        if self.keepalive_task and not self.keepalive_task.done():
            self.keepalive_task.cancel()
            try:
                await asyncio.wait_for(self.keepalive_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Close WebSocket
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.sleep(0.1)
                await asyncio.wait_for(self.ws.close(), timeout=2.0)
            except Exception:
                pass
        
        self.state = ConnectionState.DISCONNECTED


class DualStreamManager:
    """
    Manages two simultaneous Deepgram streams
    One for candidate audio, one for interviewer audio
    """
    
    def __init__(self, api_key: str, language: str = "en"):
        self.candidate_stream = DeepgramStream(api_key, StreamType.CANDIDATE, language)
        self.interviewer_stream = DeepgramStream(api_key, StreamType.INTERVIEWER, language)
        self.is_active = False
        
    async def connect_all(self) -> None:
        """Connect both streams simultaneously"""
        try:
            await asyncio.gather(
                self.candidate_stream.connect(),
                self.interviewer_stream.connect()
            )
            
            # Start keepalive tasks
            self.candidate_stream.keepalive_task = asyncio.create_task(
                self.candidate_stream.send_keepalive()
            )
            self.interviewer_stream.keepalive_task = asyncio.create_task(
                self.interviewer_stream.send_keepalive()
            )
            
            self.is_active = True
            print("✅ Both Deepgram streams ready")
            
        except Exception as e:
            print(f"❌ Failed to connect Deepgram streams: {e}")
            await self.close_all()
            raise
    
    async def close_all(self) -> None:
        """Close both streams gracefully"""
        self.is_active = False
        await asyncio.gather(
            self.candidate_stream.close(),
            self.interviewer_stream.close(),
            return_exceptions=True
        )