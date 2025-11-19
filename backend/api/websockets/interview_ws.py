"""
Interview Q&A WebSocket Handler
Real-time interview assistance with AI
"""

import asyncio
import json
import time
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import settings, validate_api_keys
from services.transcript_processor import TranscriptAccumulator
from services.ai_service import process_transcript_with_ai
from services.deepgram_manager import ConnectionState

router = APIRouter()


@router.websocket("/ws/live-interview")
async def websocket_live_interview(websocket: WebSocket):
    """
    WebSocket endpoint for live interview Q&A assistance
    Processes transcripts and generates AI responses
    """
    await websocket.accept()
    
    # Validate API key
    is_valid, error_msg = validate_api_keys(openai_key=True)
    if not is_valid:
        await websocket.send_json({
            "type": "error",
            "message": f"{error_msg}. Set in environment variables."
        })
        await websocket.close()
        return
    
    await websocket.send_json({
        "type": "connection_established",
        "message": "Q&A WebSocket ready",
        "timestamp": time.time()
    })
    
    print("\n🤖 Q&A WebSocket connected")
    
    # Connection state management
    connection_state = ConnectionState.CONNECTED
    state_lock = asyncio.Lock()
    
    async def get_state():
        async with state_lock:
            return connection_state
    
    async def set_state(new_state: ConnectionState):
        nonlocal connection_state
        async with state_lock:
            connection_state = new_state
    
    # Keepalive task
    keepalive_task = None
    should_keepalive = True
    
    async def send_render_keepalive():
        """Send periodic keepalive to prevent timeout"""
        try:
            while should_keepalive and await get_state() == ConnectionState.CONNECTED:
                await asyncio.sleep(settings.RENDER_KEEPALIVE)
                if await get_state() == ConnectionState.CONNECTED:
                    await websocket.send_json({
                        "type": "keepalive",
                        "timestamp": time.time()
                    })
        except asyncio.CancelledError:
            pass
    
    # Interview state
    transcript_accumulator = None
    prev_questions = deque(maxlen=10)
    processing_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    
    # Default settings
    user_settings = {
        "audioLanguage": "English",
        "pauseInterval": 2.0,
        "advancedQuestionDetection": False,
        "selectedResponseStyleId": "concise",
        "programmingLanguage": "Python",
        "interviewInstructions": "",
        "defaultModel": settings.DEFAULT_MODEL,
        "messageDirection": "bottom",
        "autoScroll": True
    }
    
    persona_data = None
    custom_style_prompt = None
    
    async def safe_send(data: dict) -> bool:
        """Thread-safe send with state checking"""
        state = await get_state()
        if state != ConnectionState.CONNECTED:
            return False
        try:
            async with send_lock:
                await websocket.send_json(data)
            return True
        except Exception:
            await set_state(ConnectionState.DISCONNECTING)
            return False
    
    try:
        await safe_send({"type": "ready", "message": "Q&A service ready"})
        
        # Start keepalive
        keepalive_task = asyncio.create_task(send_render_keepalive())
        
        # Main message loop
        while await get_state() == ConnectionState.CONNECTED:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=2.0
                )
                data = json.loads(message)
                
                # Handle handshake
                if data.get("type") == "client_ready":
                    await safe_send({
                        "type": "server_ack",
                        "message": "Handshake confirmed",
                        "server_time": time.time()
                    })
                    continue
                
                # Handle pong
                if data.get("type") == "pong":
                    continue
                
                # Handle initialization
                if data.get("type") == "init":
                    # Update settings
                    received_settings = data.get("settings", {})
                    user_settings.update(received_settings)
                    
                    # Initialize transcript accumulator
                    transcript_accumulator = TranscriptAccumulator(
                        pause_threshold=user_settings.get("pauseInterval", 2.0)
                    )
                    
                    # Store persona data
                    persona_data = {
                        "position": data.get("position", ""),
                        "company_name": data.get("company_name", ""),
                        "company_description": data.get("company_description", ""),
                        "job_description": data.get("job_description", ""),
                        "resume_text": data.get("resume_text", ""),
                        "resume_filename": data.get("resume_filename", "")
                    }
                    
                    custom_style_prompt = data.get("custom_style_prompt", None)
                    
                    print("=" * 60)
                    print("🎯 Q&A SESSION INITIALIZED")
                    print(f"   Position: {persona_data.get('position', 'N/A')}")
                    print(f"   Company: {persona_data.get('company_name', 'N/A')}")
                    print(f"   Style: {user_settings.get('selectedResponseStyleId')}")
                    print(f"   Model: {user_settings.get('defaultModel')}")
                    print("=" * 60)
                    
                    await safe_send({
                        "type": "connected",
                        "message": "Q&A initialized successfully"
                    })
                
                # Handle transcript
                elif data.get("type") == "transcript":
                    if not transcript_accumulator:
                        print("⚠️  Transcript received before initialization")
                        continue
                    
                    transcript = data.get("transcript", "")
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)
                    
                    print(f"📝 Transcript: {transcript[:100]}... "
                          f"(final={is_final}, speech_final={speech_final})")
                    
                    # Process transcript
                    complete_paragraph = transcript_accumulator.add_transcript(
                        transcript, 
                        is_final, 
                        speech_final
                    )
                    
                    if complete_paragraph:
                        print(f"📋 Complete paragraph: {complete_paragraph[:100]}...")
                        
                        # Skip if already processing
                        if processing_lock.locked():
                            print("⏳ Already processing, skipping...")
                            continue
                        
                        async with processing_lock:
                            # Check for duplicates
                            if any(complete_paragraph.lower() == prev.lower() 
                                   for prev in prev_questions):
                                print("⏭️  Skipping duplicate question")
                                continue
                            
                            print(f"🔍 Processing with AI...")
                            
                            # Process with AI
                            result = await process_transcript_with_ai(
                                complete_paragraph, 
                                user_settings, 
                                persona_data,
                                custom_style_prompt
                            )
                            
                            if result["has_question"]:
                                # Add to history
                                prev_questions.append(complete_paragraph)
                                
                                # Send question
                                print(f"✅ Sending question: {result['question'][:50]}...")
                                await safe_send({
                                    "type": "question_detected",
                                    "question": result["question"]
                                })
                                
                                await asyncio.sleep(0.1)
                                
                                # Send answer
                                print(f"✅ Sending answer: {result['answer'][:50]}...")
                                await safe_send({
                                    "type": "answer_ready",
                                    "question": result["question"],
                                    "answer": result["answer"]
                                })
                                
                                print("✅ Q&A pair sent successfully")
                    
            except asyncio.TimeoutError:
                continue

    except WebSocketDisconnect:
        print("❌ Q&A WebSocket disconnected")
    except Exception as e:
        print(f"❌ Q&A error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        should_keepalive = False
        await set_state(ConnectionState.DISCONNECTED)
        
        if keepalive_task:
            keepalive_task.cancel()
        
        try:
            await websocket.close()
        except:
            pass
        
        print("🔌 Q&A WebSocket closed\n")