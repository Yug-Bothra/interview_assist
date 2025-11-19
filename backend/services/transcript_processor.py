"""
Transcript Processing Service
Handles transcript accumulation and duplicate detection
"""

import time
from typing import Optional
from collections import deque
from difflib import SequenceMatcher

from config.settings import settings


class TranscriptAccumulator:
    """
    Accumulates transcript fragments into complete paragraphs
    Detects speech pauses and prevents duplicate processing
    """
    
    def __init__(self, pause_threshold: float = None):
        self.pause_threshold = pause_threshold or settings.PAUSE_THRESHOLD
        self.current_paragraph = ""
        self.last_speech_time = 0
        self.is_speaking = False
        self.complete_paragraphs = deque(maxlen=50)
        self.min_question_length = settings.MIN_QUESTION_LENGTH
        
    def add_transcript(
        self, 
        transcript: str, 
        is_final: bool, 
        speech_final: bool
    ) -> Optional[str]:
        """
        Process incoming transcript fragment
        
        Args:
            transcript: Text from speech recognition
            is_final: Whether this is a final transcript
            speech_final: Whether speech has ended
            
        Returns:
            Complete paragraph if ready, None otherwise
        """
        current_time = time.time()
        
        if not transcript or not transcript.strip():
            return None
        
        # Process complete final transcripts immediately
        if (is_final or speech_final) and len(transcript.strip()) >= self.min_question_length:
            print(f"✅ Processing complete transcript: {transcript[:100]}...")
            
            if not self._is_duplicate(transcript.strip()):
                self.complete_paragraphs.append(transcript.strip().lower())
                return transcript.strip()
            else:
                print(f"⏭️  Skipping duplicate: {transcript[:50]}...")
                return None
        
        # Accumulate partial transcripts
        if is_final or speech_final:
            if self.current_paragraph:
                self.current_paragraph += " " + transcript.strip()
            else:
                self.current_paragraph = transcript.strip()
            
            self.last_speech_time = current_time
            self.is_speaking = True
        
        # Check for pause completion
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
        """
        Check if text is too similar to recent paragraphs
        
        Args:
            text: Text to check
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if duplicate detected
        """
        text_lower = text.lower().strip()
        
        for prev in self.complete_paragraphs:
            similarity = SequenceMatcher(None, text_lower, prev).ratio()
            if similarity > threshold:
                return True
        
        return False
    
    def force_complete(self) -> Optional[str]:
        """
        Force completion of current paragraph
        
        Returns:
            Current paragraph if valid, None otherwise
        """
        if self.current_paragraph and len(self.current_paragraph) >= self.min_question_length:
            complete_text = self.current_paragraph.strip()
            self.current_paragraph = ""
            self.is_speaking = False
            return complete_text
        return None
    
    def reset(self):
        """Reset accumulator state"""
        self.current_paragraph = ""
        self.last_speech_time = 0
        self.is_speaking = False