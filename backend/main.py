"""
Unified Interview Assistant Backend
LEFT PANEL: Deepgram dual-stream (Interviewer + Candidate transcripts display)
RIGHT PANEL: Original Whisper + OpenAI Q&A (100% unchanged)
"""

import asyncio
import json
import os
import time
import re
from typing import Optional, Dict, Any
from collections import deque
from io import BytesIO
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import websockets
from websockets.exceptions import ConnectionClosed

from audio_utils import ContinuousAudioRecorder

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("❌ DEEPGRAM_API_KEY not found")

app = FastAPI(title="Unified Interview Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
WHISPER_PROMPT = "This is a conversation recording that may contain interview questions, casual talk, or technical discussions."
KEEPALIVE_INTERVAL = 5

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"

# ============================================================================
# LEFT PANEL: DEEPGRAM CONFIGURATION
# ============================================================================

def get_deepgram_url(language="en"):
    return (
        f"wss://api.deepgram.com/v1/listen"
        f"?model=nova-2"
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

class StreamType(Enum):
    CANDIDATE = "candidate"
    INTERVIEWER = "interviewer"

class DeepgramStream:
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
        self.state = ConnectionState.CONNECTING
        
        for attempt in range(self.max_retries):
            try:
                url = get_deepgram_url(self.language)
                self.ws = await websockets.connect(
                    url,
                    extra_headers={"Authorization": f"Token {self.api_key}"},
                    ping_interval=None,
                    max_size=10_000_000,
                    close_timeout=5
                )
                
                self.state = ConnectionState.CONNECTED
                emoji = "🎤" if self.stream_type == StreamType.CANDIDATE else "💻"
                print(f"{emoji} Deepgram connected ({self.stream_type.value})")
                return
            except Exception as e:
                print(f"❌ Deepgram attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    self.state = ConnectionState.DISCONNECTED
                    raise
    
    async def send_keepalive(self) -> None:
        try:
            while not self.is_closing and self.ws and self.state == ConnectionState.CONNECTED:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                if self.ws and not self.is_closing:
                    try:
                        await self.ws.send(json.dumps({"type": "KeepAlive"}))
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass
    
    async def send_audio(self, audio_data: bytes) -> bool:
        if not self.ws or self.is_closing or self.state != ConnectionState.CONNECTED:
            return False
        try:
            await self.ws.send(audio_data)
            return True
        except Exception:
            return False
    
    async def receive_transcripts(self) -> Optional[dict]:
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
        if self.is_closing:
            return
        self.is_closing = True
        self.state = ConnectionState.DISCONNECTING
        
        if self.keepalive_task and not self.keepalive_task.done():
            self.keepalive_task.cancel()
            try:
                await asyncio.wait_for(self.keepalive_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.sleep(0.1)
                await asyncio.wait_for(self.ws.close(), timeout=2.0)
            except Exception:
                pass
        self.state = ConnectionState.DISCONNECTED

class DualStreamManager:
    def __init__(self, api_key: str, language: str = "en"):
        self.candidate_stream = DeepgramStream(api_key, StreamType.CANDIDATE, language)
        self.interviewer_stream = DeepgramStream(api_key, StreamType.INTERVIEWER, language)
        self.is_active = False
        
    async def connect_all(self) -> None:
        try:
            await asyncio.gather(
                self.candidate_stream.connect(),
                self.interviewer_stream.connect()
            )
            
            self.candidate_stream.keepalive_task = asyncio.create_task(
                self.candidate_stream.send_keepalive()
            )
            self.interviewer_stream.keepalive_task = asyncio.create_task(
                self.interviewer_stream.send_keepalive()
            )
            
            self.is_active = True
            print("✅ Deepgram streams ready (LEFT PANEL)")
        except Exception as e:
            print(f"❌ Deepgram failed: {e}")
            await self.close_all()
            raise
    
    async def close_all(self) -> None:
        self.is_active = False
        await asyncio.gather(
            self.candidate_stream.close(),
            self.interviewer_stream.close(),
            return_exceptions=True
        )

# ============================================================================
# RIGHT PANEL: WHISPER + OPENAI Q&A (ORIGINAL CODE - UNCHANGED)
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

QUESTION_DETECTION_PROMPT = """You are an intelligent interview assistant that processes conversation transcripts in real-time.

Your task:
1. Analyze the incoming transcript text
2. Extract the EXACT question being asked (remove ONLY the preamble, but keep the question wording exactly as stated)
3. If a question is detected, return it in this EXACT format:
   QUESTION: [extracted question - keep original wording]
   ANSWER: [your answer]
4. If it's just casual conversation, greetings (like "hi", "hello"), or incomplete thoughts, respond with exactly: "SKIP"

Guidelines for extracting questions:
- Remove conversational preamble ONLY: "Absolutely", "I'd be happy to help", "Let's get started", "Here's a question for you", "Okay, so"
- DO NOT rephrase the question - extract it EXACTLY as the interviewer asked it
- Keep the question wording completely unchanged, including "Can you", "Could you", "Tell me", etc.
- Extract from the first question word (Can, Could, What, How, Why, etc.) to the question mark
- If multiple questions exist, extract only the most complete/recent one
- Preserve ALL technical terms, context, and original phrasing

Response format:
- If question detected: 
  QUESTION: [exact question with original wording, preamble removed]
  ANSWER: [your detailed answer]
- If no question (only greetings/fragments): SKIP

CRITICAL: Do NOT rephrase or rewrite the question. Extract it EXACTLY as spoken, removing only the conversational preamble.
"""

def clean_transcript(text: str) -> str:
    text = re.sub(r'\b(uh+|um+|ah+|like|so)\b', '', text, flags=re.I)
    return re.sub(r'\s+', ' ', text).strip()

def has_sufficient_speech(transcript: str) -> bool:
    if not transcript or len(transcript.strip()) < 5:
        return False
    noise_indicators = ['[silence]', '[noise]', '[inaudible]', '...', '..']
    if any(indicator in transcript.lower() for indicator in noise_indicators):
        return False
    return len(transcript.split()) >= 3

def is_similar_to_previous(new_text: str, prev_transcripts: deque, threshold: float = 0.9) -> bool:
    from difflib import SequenceMatcher
    new_lower = new_text.lower()
    for prev in prev_transcripts:
        if SequenceMatcher(None, new_lower, prev.lower()).ratio() > threshold:
            return True
    return False

async def transcribe_audio(audio_buffer, settings: Dict[str, Any]) -> str:
    language_map = {
        "English": "en", "Spanish": "es", "French": "fr",
        "German": "de", "Hindi": "hi", "Mandarin": "zh"
    }
    language_code = language_map.get(settings.get("audioLanguage", "English"), "en")
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                lambda: openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_buffer,
                    language=language_code,
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
            print(f"❌ Whisper error: {e}")
            return ""

async def process_transcript_with_ai(
    transcript: str,
    settings: Dict[str, Any],
    persona_data: Optional[Dict] = None,
    custom_style_prompt: Optional[str] = None
) -> Dict[str, Any]:
    try:
        response_style_id = settings.get("selectedResponseStyleId", "concise")
        
        if custom_style_prompt:
            style_prompt = custom_style_prompt
        else:
            style_config = RESPONSE_STYLES.get(response_style_id, RESPONSE_STYLES["concise"])
            style_prompt = style_config["prompt"]
        
        system_prompt = QUESTION_DETECTION_PROMPT + "\n\n" + style_prompt
        
        if persona_data:
            system_prompt += f"""

CANDIDATE CONTEXT:
- Position: {persona_data.get('position', 'N/A')}
- Company: {persona_data.get('company_name', 'N/A')}
"""
            if persona_data.get('company_description'):
                system_prompt += f"- Company Description: {persona_data.get('company_description')}\n"
            if persona_data.get('job_description'):
                system_prompt += f"- Job Description: {persona_data.get('job_description')}\n"
            if persona_data.get('resume_text'):
                system_prompt += f"\nCANDIDATE RESUME:\n{persona_data.get('resume_text')}\n"
                system_prompt += "\nIMPORTANT: Use the resume information to provide accurate, personalized answers.\n"
        
        prog_lang = settings.get("programmingLanguage", "Python")
        system_prompt += f"\n\nWhen providing code examples, use {prog_lang}."
        
        if settings.get("interviewInstructions"):
            system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{settings['interviewInstructions']}"
        
        model = settings.get("defaultModel", DEFAULT_MODEL)
        
        response = await asyncio.to_thread(
            lambda: openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript: {transcript}"}
                ],
                temperature=0.5,
                max_tokens=400,
                timeout=20
            )
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if result_text.upper() == "SKIP" or "SKIP" in result_text.upper():
            return {"has_question": False, "question": None, "answer": None}
        
        question = None
        answer = None
        
        if "QUESTION:" in result_text and "ANSWER:" in result_text:
            parts = result_text.split("ANSWER:", 1)
            question = parts[0].replace("QUESTION:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            question = transcript
            answer = result_text
        
        return {
            "has_question": True,
            "question": question,
            "answer": answer
        }
    except Exception as e:
        print(f"❌ AI error: {e}")
        return {"has_question": False, "question": None, "answer": None}

# ============================================================================
# WEBSOCKET: LEFT PANEL - DEEPGRAM DUAL-STREAM (DISPLAY ONLY)
# ============================================================================

@app.websocket("/ws/dual-transcribe")
async def websocket_dual_transcribe(websocket: WebSocket):
    """LEFT PANEL: Deepgram display for Interviewer + Candidate"""
    await websocket.accept()
    print("\n🎙️  LEFT PANEL: Deepgram connected")
    
    language = websocket.query_params.get("language", "en")
    stream_manager = DualStreamManager(DEEPGRAM_API_KEY, language)
    
    try:
        await websocket.send_json({"type": "ready", "message": "Deepgram ready"})
        await stream_manager.connect_all()
        await websocket.send_json({"type": "connected", "message": "Deepgram streams ready"})
        
        async def handle_audio():
            try:
                while stream_manager.is_active:
                    try:
                        message = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                        data = json.loads(message)
                        
                        stream_type = data.get("type")
                        audio_data = data.get("audio")
                        
                        if not audio_data or not stream_type:
                            continue
                        
                        if isinstance(audio_data, list):
                            import struct
                            audio_bytes = struct.pack(f'{len(audio_data)}h', *audio_data)
                        elif isinstance(audio_data, str):
                            import base64
                            audio_bytes = base64.b64decode(audio_data)
                        else:
                            audio_bytes = audio_data
                        
                        if stream_type == "candidate":
                            await stream_manager.candidate_stream.send_audio(audio_bytes)
                        elif stream_type == "interviewer":
                            await stream_manager.interviewer_stream.send_audio(audio_bytes)
                    except asyncio.TimeoutError:
                        continue
            except Exception as e:
                print(f"❌ Deepgram audio error: {e}")
        
        async def handle_transcripts():
            async def process_stream(stream: DeepgramStream):
                try:
                    while stream_manager.is_active and stream.state == ConnectionState.CONNECTED:
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
                                    
                                    emoji = "🎤" if stream.stream_type == StreamType.CANDIDATE else "💻"
                                    final = " [FINAL]" if transcript_data.get("speech_final") else ""
                                    print(f"{emoji} {transcript}{final}")
                except Exception as e:
                    print(f"❌ Stream error: {e}")
            
            await asyncio.gather(
                process_stream(stream_manager.candidate_stream),
                process_stream(stream_manager.interviewer_stream),
                return_exceptions=True
            )
        
        audio_task = asyncio.create_task(handle_audio())
        transcript_task = asyncio.create_task(handle_transcripts())
        
        done, pending = await asyncio.wait(
            [audio_task, transcript_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
    
    except Exception as e:
        print(f"❌ Deepgram WebSocket error: {e}")
    finally:
        await stream_manager.close_all()
        try:
            await websocket.close()
        except:
            pass
        print("🔌 LEFT PANEL: Deepgram closed\n")

# ============================================================================
# WEBSOCKET: RIGHT PANEL - WHISPER + OPENAI Q&A (ORIGINAL - 100% UNCHANGED)
# ============================================================================

@app.websocket("/ws/live-interview")
async def websocket_live_interview(websocket: WebSocket):
    """RIGHT PANEL: Original Whisper + OpenAI Q&A (100% unchanged)"""
    await websocket.accept()
    print("\n🤖 RIGHT PANEL: Q&A Copilot connected")
    
    connection_state = ConnectionState.CONNECTED
    state_lock = asyncio.Lock()
    
    async def get_state():
        async with state_lock:
            return connection_state
    
    async def set_state(new_state: ConnectionState):
        nonlocal connection_state
        async with state_lock:
            connection_state = new_state
    
    prev_transcripts = deque(maxlen=10)
    processing_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    
    settings = {
        "audioLanguage": "English",
        "pauseInterval": 2,
        "advancedQuestionDetection": False,
        "selectedResponseStyleId": "concise",
        "programmingLanguage": "Python",
        "interviewInstructions": "",
        "defaultModel": DEFAULT_MODEL,
        "messageDirection": "bottom",
        "autoScroll": True
    }
    
    persona_data = None
    custom_style_prompt = None
    recorder = None
    
    async def safe_send(data: dict) -> bool:
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
        async def handle_messages():
            nonlocal settings, persona_data, custom_style_prompt, recorder
            
            try:
                while await get_state() == ConnectionState.CONNECTED:
                    try:
                        message = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "init":
                            received_settings = data.get("settings", {})
                            settings.update(received_settings)
                            
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
                            print("🎯 RIGHT PANEL Q&A INITIALIZED")
                            print("=" * 60)
                            print(f"📋 Position: {persona_data['position']}")
                            print(f"🏢 Company: {persona_data['company_name']}")
                            print(f"📄 Resume: {'✓' if persona_data.get('resume_text') else '✗'}")
                            print(f"🎨 Style: {settings.get('selectedResponseStyleId', 'concise')}")
                            print(f"🤖 Model: {settings.get('defaultModel', DEFAULT_MODEL)}")
                            print("=" * 60)
                            
                            if not recorder:
                                try:
                                    recorder = ContinuousAudioRecorder(silence_threshold=0.3, fs=16000)
                                    recorder.start()
                                    print("✓ Audio recorder started (RIGHT PANEL)")
                                except RuntimeError as e:
                                    await safe_send({
                                        "type": "error",
                                        "message": "Audio device not found. Enable Stereo Mix."
                                    })
                                    await set_state(ConnectionState.DISCONNECTING)
                                    return
                            
                            await safe_send({"type": "connected", "message": "Q&A initialized"})
                        
                        elif data.get("type") == "ping":
                            await safe_send({"type": "pong"})
                    except asyncio.TimeoutError:
                        if await get_state() != ConnectionState.CONNECTED:
                            break
                        continue
            except WebSocketDisconnect:
                await set_state(ConnectionState.DISCONNECTING)
            except Exception as e:
                print(f"❌ Message handler error: {e}")
                await set_state(ConnectionState.DISCONNECTING)
        
        async def process_audio():
            consecutive_empty = 0
            max_empty = 15
            
            while not recorder and await get_state() == ConnectionState.CONNECTED:
                await asyncio.sleep(0.5)
            
            if not recorder:
                return
            
            while await get_state() == ConnectionState.CONNECTED:
                try:
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
                        
                        if chunk_type == "continuous_chunk.wav":
                            audio_buffer.close()
                            continue
                        
                        print(f"🎤 Processing for Q&A...")
                        start_time = time.time()
                        
                        try:
                            transcript = await transcribe_audio(audio_buffer, settings)
                        finally:
                            audio_buffer.close()

                        if not transcript:
                            continue

                        print(f"⏱️  Transcription: {time.time() - start_time:.2f}s")
                        print(f"📝 Transcript: {transcript}")

                        if is_similar_to_previous(transcript, prev_transcripts):
                            print("⏭️  Duplicate")
                            continue

                        prev_transcripts.append(transcript)

                        ai_start = time.time()
                        result = await process_transcript_with_ai(
                            transcript, 
                            settings, 
                            persona_data,
                            custom_style_prompt
                        )
                        print(f"⏱️  AI: {time.time() - ai_start:.2f}s")

                        if await get_state() != ConnectionState.CONNECTED:
                            break

                        if result["has_question"]:
                            print(f"❓ Question: {result['question']}")
                            print(f"💬 Answer: {result['answer'][:100]}...")
                            
                            if not await safe_send({
                                "type": "question_detected",
                                "question": result["question"]
                            }):
                                break
                            
                            await asyncio.sleep(0.1)
                            
                            if not await safe_send({
                                "type": "answer_ready",
                                "question": result["question"],
                                "answer": result["answer"]
                            }):
                                break
                        else:
                            print("⏭️  No question")
                except WebSocketDisconnect:
                    await set_state(ConnectionState.DISCONNECTING)
                    break
                except Exception as e:
                    if await get_state() == ConnectionState.CONNECTED:
                        print(f"❌ Audio error: {e}")
                    break
        
        await safe_send({"type": "ready", "message": "Q&A ready"})
        
        message_task = asyncio.create_task(handle_messages())
        audio_task = asyncio.create_task(process_audio())
        
        done, pending = await asyncio.wait(
            [message_task, audio_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        await set_state(ConnectionState.DISCONNECTING)
        
        for task in pending:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    except WebSocketDisconnect:
        print("❌ Q&A client disconnected")
    except Exception as e:
        print(f"❌ Q&A error: {e}")
    finally:
        await set_state(ConnectionState.DISCONNECTED)
        
        if recorder:
            try:
                recorder.stop()
                print("✓ Audio recorder stopped (RIGHT PANEL)")
            except Exception as e:
                print(f"⚠️  Recorder stop error: {e}")
        
        try:
            await websocket.close()
        except Exception:
            pass
            
        print("🔌 RIGHT PANEL: Q&A closed\n")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Unified Interview Assistant API",
        "left_panel": "Deepgram dual-stream display",
        "right_panel": "Whisper + OpenAI Q&A"
    }

@app.get("/api/models/status")
async def get_model_status():
    return {
        "default_provider": DEFAULT_MODEL,
        "available_providers": {"gpt-4o-mini": True, "gpt-4o": True}
    }

@app.post("/api/models/set-default")
async def set_default_model(data: dict):
    return {"success": True, "provider": data.get("provider", DEFAULT_MODEL)}

@app.post("/api/models/set-coding")
async def set_coding_model(data: dict):
    return {"success": True, "provider": data.get("provider", DEFAULT_MODEL)}

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 UNIFIED INTERVIEW ASSISTANT BACKEND")
    print("=" * 70)
    print()
    print("LEFT PANEL (Deepgram Display):")
    print("  📡 WebSocket: ws://localhost:8000/ws/dual-transcribe")
    print("  🎤 Shows: Candidate (Microphone)")
    print("  💻 Shows: Interviewer (System Audio)")
    print("  ✓ Real-time transcription display only")
    print()
    print("RIGHT PANEL (Original Whisper + OpenAI Q&A):")
    print("  📡 WebSocket: ws://localhost:8000/ws/live-interview")
    print("  🎙️  Captures: System audio via audio_utils.py")
    print("  🤖 Processes: Whisper transcription → OpenAI Q&A")
    print("  ✓ Question detection + Answer generation")
    print("  ✓ Resume-aware responses")
    print()
    print("🔑 APIs:")
    print(f"  Deepgram: {'✅ Ready' if DEEPGRAM_API_KEY else '❌ Missing'}")
    print(f"  OpenAI: {'✅ Ready' if openai.api_key else '❌ Missing'}")
    print()
    print("📊 Endpoints:")
    print("  Health: http://localhost:8000/health")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 70)
    print("✨ LEFT shows transcripts | RIGHT processes Q&A")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )