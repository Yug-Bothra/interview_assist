"""
Render-Compatible Interview Assistant Backend
Audio capture: 100% browser-based (no server audio devices needed)
LEFT PANEL: Deepgram dual-stream transcription display
RIGHT PANEL: Q&A with Deepgram transcripts (no Whisper, no audio_utils.py)
"""

import asyncio
import json
import os
import time
from typing import Optional, Dict, Any
from collections import deque
from enum import Enum
from difflib import SequenceMatcher

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import websockets
from websockets.exceptions import ConnectionClosed

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("❌ DEEPGRAM_API_KEY not found")

app = FastAPI(title="Interview Assistant API - Render Compatible")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
KEEPALIVE_INTERVAL = 5

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"

# ============================================================================
# DEEPGRAM CONFIGURATION
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
            print("✅ Deepgram streams ready")
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
# TRANSCRIPT ACCUMULATOR FOR Q&A
# ============================================================================

class TranscriptAccumulator:
    """Accumulates Deepgram transcripts and detects complete questions"""
    
    def __init__(self, pause_threshold: float = 2.0):
        self.pause_threshold = pause_threshold
        self.current_paragraph = ""
        self.last_speech_time = 0
        self.is_speaking = False
        self.complete_paragraphs = deque(maxlen=50)
        self.min_question_length = 10
        
    def add_transcript(self, transcript: str, is_final: bool, speech_final: bool) -> Optional[str]:
        """Add transcript chunk and return complete paragraph if pause detected"""
        current_time = time.time()
        
        if not transcript or not transcript.strip():
            return None
        
        if is_final or speech_final:
            if self.current_paragraph:
                self.current_paragraph += " " + transcript.strip()
            else:
                self.current_paragraph = transcript.strip()
            
            self.last_speech_time = current_time
            self.is_speaking = True
        
        if self.is_speaking and self.current_paragraph:
            time_since_last_speech = current_time - self.last_speech_time
            
            if time_since_last_speech >= self.pause_threshold:
                complete_text = self.current_paragraph.strip()
                
                if len(complete_text) >= self.min_question_length:
                    if not self._is_duplicate(complete_text):
                        self.complete_paragraphs.append(complete_text.lower())
                        self.current_paragraph = ""
                        self.is_speaking = False
                        return complete_text
                
                self.current_paragraph = ""
                self.is_speaking = False
        
        return None
    
    def _is_duplicate(self, text: str, threshold: float = 0.85) -> bool:
        """Check if text is too similar to recent paragraphs"""
        text_lower = text.lower().strip()
        
        for prev in self.complete_paragraphs:
            similarity = SequenceMatcher(None, text_lower, prev).ratio()
            if similarity > threshold:
                return True
        
        return False
    
    def force_complete(self) -> Optional[str]:
        """Force completion of current paragraph"""
        if self.current_paragraph and len(self.current_paragraph) >= self.min_question_length:
            complete_text = self.current_paragraph.strip()
            self.current_paragraph = ""
            self.is_speaking = False
            return complete_text
        return None

# ============================================================================
# Q&A PROCESSING
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
- Remove conversational preamble ONLY
- DO NOT rephrase the question - extract it EXACTLY as asked
- Keep the question wording completely unchanged
- Extract from the first question word to the question mark
- Preserve ALL technical terms, context, and original phrasing

Response format:
- If question detected: 
  QUESTION: [exact question with original wording]
  ANSWER: [your detailed answer]
- If no question: SKIP

CRITICAL: Do NOT rephrase or rewrite the question. Extract it EXACTLY as spoken.
"""

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
# WEBSOCKET: DEEPGRAM DUAL-STREAM (LEFT PANEL)
# ============================================================================

@app.websocket("/ws/dual-transcribe")
async def websocket_dual_transcribe(websocket: WebSocket):
    """Deepgram display for Interviewer + Candidate"""
    await websocket.accept()
    print("\n🎙️ Deepgram connected")
    
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
                print(f"❌ Audio error: {e}")
        
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
        print(f"❌ Deepgram error: {e}")
    finally:
        await stream_manager.close_all()
        try:
            await websocket.close()
        except:
            pass
        print("🔌 Deepgram closed\n")

# ============================================================================
# WEBSOCKET: Q&A WITH DEEPGRAM TRANSCRIPTS (RIGHT PANEL)
# ============================================================================

@app.websocket("/ws/live-interview")
async def websocket_live_interview(websocket: WebSocket):
    """Q&A Copilot - processes Deepgram transcripts from frontend"""
    await websocket.accept()
    print("\n🤖 Q&A Copilot connected")
    
    connection_state = ConnectionState.CONNECTED
    state_lock = asyncio.Lock()
    
    async def get_state():
        async with state_lock:
            return connection_state
    
    async def set_state(new_state: ConnectionState):
        nonlocal connection_state
        async with state_lock:
            connection_state = new_state
    
    transcript_accumulator = None
    prev_questions = deque(maxlen=10)
    processing_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    
    settings = {
        "audioLanguage": "English",
        "pauseInterval": 2.0,
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
        await safe_send({"type": "ready", "message": "Q&A ready"})
        
        while await get_state() == ConnectionState.CONNECTED:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
                data = json.loads(message)
                
                if data.get("type") == "init":
                    received_settings = data.get("settings", {})
                    settings.update(received_settings)
                    
                    transcript_accumulator = TranscriptAccumulator(
                        pause_threshold=settings.get("pauseInterval", 2.0)
                    )
                    
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
                    print("🎯 Q&A INITIALIZED")
                    print("=" * 60)
                    print(f"⏱️  Pause threshold: {transcript_accumulator.pause_threshold}s")
                    print(f"📋 Position: {persona_data['position']}")
                    print(f"🤖 Model: {settings.get('defaultModel', DEFAULT_MODEL)}")
                    print("=" * 60)
                    
                    await safe_send({"type": "connected", "message": "Q&A initialized"})
                
                elif data.get("type") == "transcript":
                    if not transcript_accumulator:
                        continue
                    
                    transcript = data.get("transcript", "")
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)
                    
                    complete_paragraph = transcript_accumulator.add_transcript(
                        transcript, 
                        is_final, 
                        speech_final
                    )
                    
                    if complete_paragraph:
                        if processing_lock.locked():
                            print("⏭️  Skipping - already processing")
                            continue
                        
                        async with processing_lock:
                            print(f"\n📝 Complete paragraph detected:")
                            print(f"   {complete_paragraph[:100]}...")
                            
                            if any(complete_paragraph.lower() == prev.lower() for prev in prev_questions):
                                print("⏭️  Duplicate question skipped")
                                continue
                            
                            print("🤖 Processing with AI...")
                            result = await process_transcript_with_ai(
                                complete_paragraph, 
                                settings, 
                                persona_data,
                                custom_style_prompt
                            )
                            
                            if result["has_question"]:
                                prev_questions.append(complete_paragraph)
                                
                                print(f"❓ Question: {result['question']}")
                                print(f"💬 Answer: {result['answer'][:100]}...")
                                
                                await safe_send({
                                    "type": "question_detected",
                                    "question": result["question"]
                                })
                                
                                await asyncio.sleep(0.1)
                                
                                await safe_send({
                                    "type": "answer_ready",
                                    "question": result["question"],
                                    "answer": result["answer"]
                                })
                            else:
                                print("⏭️  No question detected")
                
                elif data.get("type") == "ping":
                    await safe_send({"type": "pong"})
                    
            except asyncio.TimeoutError:
                if transcript_accumulator and transcript_accumulator.current_paragraph:
                    current_time = time.time()
                    time_since_last = current_time - transcript_accumulator.last_speech_time
                    
                    if time_since_last >= transcript_accumulator.pause_threshold:
                        complete_paragraph = transcript_accumulator.force_complete()
                        
                        if complete_paragraph and not processing_lock.locked():
                            async with processing_lock:
                                print(f"\n⏰ Force completing paragraph (timeout)")
                                
                                if not any(complete_paragraph.lower() == prev.lower() for prev in prev_questions):
                                    result = await process_transcript_with_ai(
                                        complete_paragraph, 
                                        settings, 
                                        persona_data,
                                        custom_style_prompt
                                    )
                                    
                                    if result["has_question"]:
                                        prev_questions.append(complete_paragraph)
                                        
                                        await safe_send({
                                            "type": "question_detected",
                                            "question": result["question"]
                                        })
                                        
                                        await asyncio.sleep(0.1)
                                        
                                        await safe_send({
                                            "type": "answer_ready",
                                            "question": result["question"],
                                            "answer": result["answer"]
                                        })
                continue

    except WebSocketDisconnect:
        print("❌ Q&A disconnected")
    except Exception as e:
        print(f"❌ Q&A error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await set_state(ConnectionState.DISCONNECTED)
        try:
            await websocket.close()
        except:
            pass
        print("🔌 Q&A closed\n")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Interview Assistant API - Render Compatible",
        "audio_capture": "browser-based",
        "server_audio": "not required"
    }

@app.get("/api/models/status")
async def get_model_status():
    return {
        "default_provider": DEFAULT_MODEL,
        "available_providers": {"gpt-4o-mini": True, "gpt-4o": True}
    }

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 INTERVIEW ASSISTANT BACKEND (RENDER COMPATIBLE)")
    print("=" * 70)
    print("✅ Audio capture: Browser-based (100%)")
    print("✅ No server audio devices required")
    print("✅ Deepgram: Real-time transcription")
    print("✅ OpenAI: Q&A generation")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )