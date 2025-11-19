"""
Deepgram WebSocket Handler
Dual-stream audio transcription
"""

import asyncio
import json
import time
import base64
import struct

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import settings, validate_api_keys
from services.deepgram_manager import DualStreamManager, ConnectionState

router = APIRouter()


@router.websocket("/ws/dual-transcribe")
async def websocket_dual_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for dual-stream Deepgram transcription
    Handles separate audio streams for candidate and interviewer
    """
    await websocket.accept()
    
    # Validate API key
    is_valid, error_msg = validate_api_keys(deepgram=True)
    if not is_valid:
        await websocket.send_json({
            "type": "error",
            "message": f"{error_msg}. Set in environment variables."
        })
        await websocket.close()
        return
    
    await websocket.send_json({
        "type": "connection_established",
        "message": "Deepgram WebSocket ready",
        "timestamp": time.time()
    })
    
    print("\n🎙️  Deepgram WebSocket connected")
    
    # Get language from query params
    language = websocket.query_params.get("language", "en")
    stream_manager = DualStreamManager(settings.DEEPGRAM_API_KEY, language)
    
    # Keepalive task
    keepalive_task = None
    should_keepalive = True
    
    async def send_render_keepalive():
        """Send periodic keepalive to prevent timeout"""
        try:
            while should_keepalive:
                await asyncio.sleep(settings.RENDER_KEEPALIVE)
                if should_keepalive:
                    try:
                        await websocket.send_json({
                            "type": "keepalive",
                            "timestamp": time.time()
                        })
                        print(f"🏓 Keepalive sent")
                    except Exception as e:
                        print(f"❌ Keepalive failed: {e}")
                        break
        except asyncio.CancelledError:
            print("⏹️  Keepalive cancelled")
    
    try:
        # Initialize connection
        await websocket.send_json({"type": "ready", "message": "Initializing Deepgram"})
        await stream_manager.connect_all()
        
        # Start keepalive
        keepalive_task = asyncio.create_task(send_render_keepalive())
        
        await websocket.send_json({
            "type": "connected",
            "message": "Deepgram streams active"
        })
        
        async def handle_audio():
            """Process incoming audio data"""
            try:
                while stream_manager.is_active:
                    try:
                        message = await asyncio.wait_for(
                            websocket.receive_text(), 
                            timeout=0.1
                        )
                        data = json.loads(message)
                        
                        # Handle handshake
                        if data.get("type") == "client_ready":
                            await websocket.send_json({
                                "type": "server_ack",
                                "message": "Handshake confirmed",
                                "server_time": time.time()
                            })
                            continue
                        
                        # Handle pong
                        if data.get("type") == "pong":
                            continue
                        
                        # Process audio
                        stream_type = data.get("type")
                        audio_data = data.get("audio")
                        
                        if not audio_data or not stream_type:
                            continue
                        
                        # Convert audio data to bytes
                        if isinstance(audio_data, list):
                            audio_bytes = struct.pack(f'{len(audio_data)}h', *audio_data)
                        elif isinstance(audio_data, str):
                            audio_bytes = base64.b64decode(audio_data)
                        else:
                            audio_bytes = audio_data
                        
                        # Send to appropriate stream
                        if stream_type == "candidate":
                            await stream_manager.candidate_stream.send_audio(audio_bytes)
                        elif stream_type == "interviewer":
                            await stream_manager.interviewer_stream.send_audio(audio_bytes)
                            
                    except asyncio.TimeoutError:
                        continue
                        
            except Exception as e:
                print(f"❌ Audio handling error: {e}")
        
        async def handle_transcripts():
            """Process and forward transcripts from both streams"""
            async def process_stream(stream):
                try:
                    while (stream_manager.is_active and 
                           stream.state == ConnectionState.CONNECTED):
                        
                        transcript_data = await stream.receive_transcripts()
                        
                        if not transcript_data:
                            await asyncio.sleep(0.01)
                            continue
                        
                        if transcript_data.get("type") == "Results":
                            channel = transcript_data.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            
                            if alternatives and len(alternatives) > 0:
                                transcript = alternatives[0].get("transcript", "")
                                
                                if transcript.strip():
                                    response = {
                                        "type": "transcript",
                                        "stream": stream.stream_type.value,
                                        "transcript": transcript,
                                        "is_final": transcript_data.get("is_final", False),
                                        "speech_final": transcript_data.get("speech_final", False)
                                    }
                                    await websocket.send_json(response)
                                    
                except Exception as e:
                    print(f"❌ Transcript stream error: {e}")
            
            # Process both streams concurrently
            await asyncio.gather(
                process_stream(stream_manager.candidate_stream),
                process_stream(stream_manager.interviewer_stream),
                return_exceptions=True
            )
        
        # Run both handlers concurrently
        audio_task = asyncio.create_task(handle_audio())
        transcript_task = asyncio.create_task(handle_transcripts())
        
        # Wait for either to complete
        done, pending = await asyncio.wait(
            [audio_task, transcript_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
    
    except WebSocketDisconnect:
        print("❌ Deepgram WebSocket disconnected")
    except Exception as e:
        print(f"❌ Deepgram error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        should_keepalive = False
        if keepalive_task:
            keepalive_task.cancel()
        
        await stream_manager.close_all()
        
        try:
            await websocket.close()
        except:
            pass
        
        print("🔌 Deepgram WebSocket closed\n")