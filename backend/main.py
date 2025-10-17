import asyncio
import json
import os
import time
import re
from typing import Optional, Dict, Any
from collections import deque
from io import BytesIO
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
from audio_utils import ContinuousAudioRecorder

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Interview Assistant API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONNECTION STATE MANAGEMENT
# ============================================================================

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
WHISPER_PROMPT = "This is a conversation recording that may contain interview questions, casual talk, or technical discussions."

# ============================================================================
# RESPONSE STYLE PROMPTS
# ============================================================================

RESPONSE_STYLES = {
    "concise": {
        "name": "Concise Professional",
        "prompt": """You are a concise interview assistant. Provide brief, professional answers in 2-3 sentences.
Focus on the core information without elaboration. Be direct and efficient."""
    },
    "detailed": {
        "name": "Detailed Professional",
        "prompt": """You are a detailed interview assistant. Provide comprehensive answers with:
- Clear explanation of the concept
- Relevant examples from experience
- Practical insights
Keep responses around 150 words, professional and well-structured."""
    },
    "storytelling": {
        "name": "Storytelling",
        "prompt": """You are an engaging interview assistant using storytelling techniques.
Structure answers using STAR format when appropriate:
- Situation: Set the context
- Task: Describe the challenge
- Action: Explain what you did
- Result: Share the outcome
Make responses compelling and memorable while remaining professional."""
    },
    "technical": {
        "name": "Technical Expert",
        "prompt": """You are a technical interview expert. Provide in-depth technical answers:
- Explain concepts clearly with proper terminology
- Include code examples when relevant
- Discuss trade-offs and best practices
Be thorough but avoid unnecessary jargon."""
    }
}

# ============================================================================
# QUESTION DETECTION PROMPT
# ============================================================================

QUESTION_DETECTION_PROMPT = """You are an intelligent interview assistant that processes conversation transcripts in real-time.

Your task:
1. Analyze the incoming transcript text
2. Identify if there's a clear question being asked (technical, behavioral, HR-related, personal, or general questions)
3. If a question is detected, provide a concise, natural answer as if you're the candidate being interviewed
4. If it's just casual conversation, greetings (like "hi", "hello"), or incomplete thoughts, respond with exactly: "SKIP"

Guidelines:
- ANY sentence with a question mark (?) should be treated as a question
- ANY sentence starting with question words (what, why, how, when, where, who, which, can, could, would, should, do, does, did, is, are) is likely a question
- Commands like "tell me about", "describe your", "explain how" are questions
- Answer ALL types of questions: technical, behavioral, personal, hypothetical, or general
- If multiple questions are in the text, answer the most recent/complete one
- Only respond "SKIP" for pure greetings without questions or incomplete fragments

Response format:
- If ANY question detected: Provide the answer directly (never say "SKIP")
- If no question (only greetings/fragments): Respond with exactly "SKIP"

Examples:
- "What are your hobbies?" → Answer it
- "Tell me about yourself" → Answer it (command form question)
- "How do you handle stress?" → Answer it
- "Hi" → SKIP
- "Hello there" → SKIP
- "Invest your time in" → SKIP (incomplete)
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_transcript(text: str) -> str:
    """Clean up transcript by removing filler words and extra spaces"""
    text = re.sub(r'\b(uh+|um+|ah+|like|so)\b', '', text, flags=re.I)
    return re.sub(r'\s+', ' ', text).strip()

def is_similar_to_previous(new_text: str, prev_transcripts: deque, threshold: float = 0.9) -> bool:
    """Check if this transcript is too similar to recent ones (avoid duplicates)"""
    from difflib import SequenceMatcher
    new_lower = new_text.lower()
    for prev in prev_transcripts:
        if SequenceMatcher(None, new_lower, prev.lower()).ratio() > threshold:
            return True
    return False

def has_sufficient_speech(transcript: str) -> bool:
    """
    Check if transcript has sufficient speech content.
    Returns False for empty, very short, or low-confidence transcripts.
    """
    if not transcript or len(transcript.strip()) < 5:
        return False
    
    # Check if it's mostly silence indicators or noise
    noise_indicators = ['[silence]', '[noise]', '[inaudible]', '...', '..']
    if any(indicator in transcript.lower() for indicator in noise_indicators):
        return False
    
    # Check word count - at least 3 words for meaningful content
    word_count = len(transcript.split())
    if word_count < 3:
        return False
    
    return True

# ============================================================================
# AI PROCESSING FUNCTIONS
# ============================================================================

async def transcribe_audio(audio_buffer) -> str:
    """Transcribe audio using Whisper with retry logic"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                lambda: openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_buffer,
                    language="en",
                    prompt=WHISPER_PROMPT,
                    temperature=0.0
                )
            )
            transcript = clean_transcript(response.text.strip())
            
            if has_sufficient_speech(transcript):
                return transcript
            return ""
            
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                audio_buffer.seek(0)
                continue
            return ""
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

async def process_transcript_with_ai(
    transcript: str,
    settings: Dict[str, Any],
    persona_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Send transcript to OpenAI to intelligently extract question and generate answer.
    Returns: {"has_question": bool, "question": str, "answer": str}
    """
    try:
        # Build system prompt based on settings
        response_style = settings.get("selectedResponseStyleId", "concise")
        style_config = RESPONSE_STYLES.get(response_style, RESPONSE_STYLES["concise"])
        
        system_prompt = QUESTION_DETECTION_PROMPT + "\n\n" + style_config["prompt"]
        
        if persona_data:
            system_prompt += f"""

CANDIDATE CONTEXT:
- Position: {persona_data.get('position', 'N/A')}
- Company: {persona_data.get('company_name', 'N/A')}
"""
        
        prog_lang = settings.get("programmingLanguage", "Python")
        system_prompt += f"\n\nWhen providing code examples, use {prog_lang}."
        
        if settings.get("interviewInstructions"):
            system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{settings['interviewInstructions']}"
        
        response = await asyncio.to_thread(
            lambda: openai.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript: {transcript}"}
                ],
                temperature=0.5,
                max_tokens=300,
                timeout=20
            )
        )
        
        answer = response.choices[0].message.content.strip()
        
        if answer.upper() == "SKIP" or "SKIP" in answer.upper():
            return {"has_question": False, "question": None, "answer": None}
        
        return {
            "has_question": True,
            "question": transcript,
            "answer": answer
        }
        
    except openai.RateLimitError:
        await asyncio.sleep(1)
        return {"has_question": False, "question": None, "answer": None}
    except openai.Timeout:
        return {"has_question": False, "question": None, "answer": None}
    except Exception as e:
        print(f"❌ AI processing error: {e}")
        return {"has_question": False, "question": None, "answer": None}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Interview Assistant API is running"}

@app.get("/api/models/status")
async def get_model_status():
    """Get current model configuration"""
    return {
        "default_provider": DEFAULT_MODEL,
        "available_providers": {"gpt-4o-mini": True, "gpt-4o": True}
    }

# ============================================================================
# WEBSOCKET ENDPOINT - FULLY FIXED WITH PROPER STATE MANAGEMENT
# ============================================================================

@app.websocket("/ws/live-interview")
async def websocket_live_interview(websocket: WebSocket):
    """
    WebSocket endpoint for real-time interview assistance.
    FULLY FIXED: Proper state management, connection stability, and graceful cleanup
    """
    await websocket.accept()
    print("✅ WebSocket connection accepted")
    
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
            print(f"🔄 Connection state: {new_state.value}")
    
    # Session data
    prev_transcripts = deque(maxlen=10)
    processing_lock = asyncio.Lock()
    send_lock = asyncio.Lock()  # Prevent concurrent sends
    
    settings = {
        "selectedResponseStyleId": "concise",
        "programmingLanguage": "Python",
        "interviewInstructions": ""
    }
    persona_data = None
    recorder = None
    
    print("✓ WebSocket client connected - initializing...")
    
    # Initialize audio recorder
    try:
        recorder = ContinuousAudioRecorder(silence_threshold=0.3, fs=16000)
        recorder.start()
        print("✓ Audio recorder started")
    except RuntimeError as e:
        await websocket.send_json({
            "type": "error",
            "message": "Audio device not found. Please enable Stereo Mix in Windows Sound Settings."
        })
        await websocket.close()
        return
    
    # Safe send function with state checking
    async def safe_send(data: dict) -> bool:
        """Send data only if connection is active"""
        state = await get_state()
        if state != ConnectionState.CONNECTED:
            return False
        
        try:
            async with send_lock:
                await websocket.send_json(data)
            return True
        except Exception as e:
            print(f"❌ Send error: {e}")
            await set_state(ConnectionState.DISCONNECTING)
            return False
    
    try:
        # Send ready message
        await safe_send({
            "type": "ready",
            "message": "Backend ready for audio processing"
        })
        
        async def handle_incoming_messages():
            """Handle control messages from client"""
            try:
                while await get_state() == ConnectionState.CONNECTED:
                    try:
                        # Use shorter timeout for more responsive disconnect detection
                        message = await asyncio.wait_for(
                            websocket.receive_text(), 
                            timeout=2.0
                        )
                        data = json.loads(message)
                        
                        if data.get("type") == "init":
                            nonlocal settings, persona_data
                            settings = data.get("settings", settings)
                            persona_data = {
                                "position": data.get("position", ""),
                                "company_name": data.get("company_name", "")
                            }
                            print(f"✓ Session initialized with settings: {settings}")
                            
                            await safe_send({
                                "type": "connected",
                                "message": "Session initialized"
                            })
                        
                        elif data.get("type") == "ping":
                            # Respond to keepalive
                            await safe_send({"type": "pong"})
                            
                    except asyncio.TimeoutError:
                        # Check connection health
                        if await get_state() != ConnectionState.CONNECTED:
                            break
                        continue
                        
            except WebSocketDisconnect:
                print("❌ Client disconnected from message handler")
                await set_state(ConnectionState.DISCONNECTING)
            except Exception as e:
                print(f"❌ Message handler error: {e}")
                await set_state(ConnectionState.DISCONNECTING)
        
        async def process_audio():
            """Process audio chunks with proper state management"""
            consecutive_empty = 0
            max_empty = 15
            
            while await get_state() == ConnectionState.CONNECTED:
                try:
                    # Check state before processing
                    if await get_state() != ConnectionState.CONNECTED:
                        break
                    
                    async with processing_lock:
                        audio_buffer = recorder.get_audio_chunk()
                        
                        if audio_buffer is None:
                            consecutive_empty += 1
                            if consecutive_empty >= max_empty:
                                consecutive_empty = 0
                            await asyncio.sleep(0.2)
                            continue
                        
                        consecutive_empty = 0
                        chunk_type = getattr(audio_buffer, 'name', 'unknown')
                        
                        # Skip continuous chunks (no speech detected)
                        if chunk_type == "continuous_chunk.wav":
                            print("⏭️  Skipping continuous chunk")
                            audio_buffer.close()
                            continue
                        
                        print(f"🎤 Processing speech segment...")
                        start_time = time.time()
                        
                        try:
                            transcript = await transcribe_audio(audio_buffer)
                        finally:
                            audio_buffer.close()

                        if not transcript:
                            print("⏭️  No valid speech")
                            continue

                        print(f"⏱️  Transcription: {time.time() - start_time:.2f}s")
                        print(f"📝 Transcript: {transcript}")

                        if is_similar_to_previous(transcript, prev_transcripts):
                            print("⏭️  Duplicate transcript")
                            continue

                        prev_transcripts.append(transcript)

                        # Send transcript (check state first)
                        if not await safe_send({
                            "type": "transcript",
                            "text": transcript
                        }):
                            break

                        # Process with AI
                        ai_start = time.time()
                        result = await process_transcript_with_ai(
                            transcript, settings, persona_data
                        )
                        print(f"⏱️  AI processing: {time.time() - ai_start:.2f}s")

                        if await get_state() != ConnectionState.CONNECTED:
                            break

                        if result["has_question"]:
                            print(f"❓ Question: {result['question']}")
                            print(f"💬 Answer: {result['answer'][:100]}...")
                            
                            # Send question detection
                            if not await safe_send({
                                "type": "question_detected",
                                "question": result["question"]
                            }):
                                break
                            
                            # Small delay between messages
                            await asyncio.sleep(0.1)
                            
                            # Send answer
                            if not await safe_send({
                                "type": "answer_ready",
                                "question": result["question"],
                                "answer": result["answer"]
                            }):
                                break
                        else:
                            print("⏭️  No question detected")
                            
                except WebSocketDisconnect:
                    print("❌ Client disconnected from audio processor")
                    await set_state(ConnectionState.DISCONNECTING)
                    break
                except Exception as e:
                    if await get_state() == ConnectionState.CONNECTED:
                        print(f"❌ Audio processing error: {e}")
                    break
        
        # Run both tasks with proper cancellation
        message_task = asyncio.create_task(handle_incoming_messages())
        audio_task = asyncio.create_task(process_audio())
        
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [message_task, audio_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Update state and cancel remaining tasks
        await set_state(ConnectionState.DISCONNECTING)
        
        for task in pending:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        await set_state(ConnectionState.DISCONNECTED)
        
        # Cleanup recorder
        if recorder:
            try:
                recorder.stop()
                print("✓ Audio recorder stopped")
            except Exception as e:
                print(f"⚠️  Recorder stop error: {e}")
        
        # Close websocket gracefully
        try:
            await websocket.close()
        except Exception:
            pass
            
        print("🔌 WebSocket connection closed cleanly")

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Interview Assistant Backend Starting...")
    print("=" * 60)
    print(f"Model: {DEFAULT_MODEL}")
    print("WebSocket: ws://127.0.0.1:8000/ws/live-interview")
    print("Health Check: http://127.0.0.1:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )