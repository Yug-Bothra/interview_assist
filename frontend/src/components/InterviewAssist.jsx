import React, { useState, useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "./Auth/AuthContext";
import {
  Mic,
  Download,
  Settings,
  ArrowLeft,
  Pause,
  Play,
  Volume2,
  X,
  AlertCircle,
  MicOff
} from "lucide-react";
import { jsPDF } from "jspdf";

// ============================================================================
// STREAMING COMPONENTS
// ============================================================================

function StreamingText({ text, isComplete, className = "" }) {
  const [displayedWords, setDisplayedWords] = useState([]);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);

  useEffect(() => {
    if (isComplete) {
      setDisplayedWords(text.split(' '));
      return;
    }

    const words = text.split(' ');
    if (currentWordIndex < words.length) {
      const timer = setTimeout(() => {
        setDisplayedWords(words.slice(0, currentWordIndex + 1));
        setCurrentWordIndex(currentWordIndex + 1);
      }, 80);

      return () => clearTimeout(timer);
    }
  }, [text, currentWordIndex, isComplete]);

  return (
    <p className={`whitespace-pre-wrap leading-relaxed ${className}`}>
      {displayedWords.join(' ')}
      {!isComplete && currentWordIndex < text.split(' ').length && (
        <span className="inline-block w-1 h-4 bg-blue-400 ml-1 animate-pulse"></span>
      )}
    </p>
  );
}

function StreamingAnswer({ text, isComplete }) {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (isComplete) {
      setDisplayedText(text);
      return;
    }

    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText(text.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 20);

      return () => clearTimeout(timer);
    }
  }, [text, currentIndex, isComplete]);

  return (
    <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">
      {displayedText}
      {!isComplete && currentIndex < text.length && (
        <span className="inline-block w-1 h-4 bg-green-400 ml-1 animate-pulse"></span>
      )}
    </p>
  );
}

// ============================================================================
// QA LIST
// ============================================================================

function QAList({ qaList }) {
  return (
    <div className="space-y-4">
      {qaList.map((item, index) => {
        const questionNumber = index + 1;

        return (
          <div key={item.id} className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="mb-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold text-purple-400 bg-purple-900/30 px-2 py-1 rounded">
                  ❓ QUESTION #{questionNumber}
                </span>
              </div>
              <p className="text-gray-200 font-medium leading-relaxed">{item.question}</p>
            </div>

            <div className="border-t border-gray-800 pt-3 mt-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold text-green-400 bg-green-900/30 px-2 py-1 rounded">💬 ANSWER</span>
              </div>
              <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                {item.answer}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function InterviewAssist() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, loading } = useAuth();

  // Persona & Settings State
  const [personaId] = useState(
    location.state?.personaId || localStorage.getItem("selectedPersona") || null
  );
  const [personaData] = useState(() => {
    if (location.state?.personaData) return location.state.personaData;
    const saved = localStorage.getItem("selectedPersonaData");
    try {
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [domain] = useState(
    location.state?.domain || localStorage.getItem("selectedDomain") || ""
  );
  const [settings] = useState(() => {
    if (location.state?.settings) return location.state.settings;
    const saved = localStorage.getItem("copilotSettings");
    try {
      return saved
        ? JSON.parse(saved)
        : {
          responseStyle: "professional",
          audioLanguage: "English",
          pauseInterval: 2.0,
          advancedQuestionDetection: true,
          messageDirection: "bottom",
          autoScroll: true,
          programmingLanguage: "Python",
          selectedResponseStyleId: "concise",
          defaultModel: "gpt-4o-mini"
        };
    } catch {
      return {
        responseStyle: "professional",
        audioLanguage: "English",
        pauseInterval: 2.0,
        advancedQuestionDetection: true,
        messageDirection: "bottom",
        autoScroll: true,
        programmingLanguage: "Python",
        selectedResponseStyleId: "concise",
        defaultModel: "gpt-4o-mini"
      };
    }
  });

  // Component States
  const [isRecording, setIsRecording] = useState(false);
  const [qaList, setQaList] = useState([]);
  const [candidateTranscript, setCandidateTranscript] = useState([]);
  const [interviewerTranscript, setInterviewerTranscript] = useState([]);
  const [activeView, setActiveView] = useState("interviewer");
  const [deepgramStatus, setDeepgramStatus] = useState("Ready");
  const [qaStatus, setQaStatus] = useState("Ready");
  const [showSettings, setShowSettings] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [showTabModal, setShowTabModal] = useState(false);
  const [tabAudioError, setTabAudioError] = useState("");

  // Current Q&A states
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStreamingComplete, setIsStreamingComplete] = useState(false);

  // Current paragraph building states
  const [currentCandidateParagraph, setCurrentCandidateParagraph] = useState("");
  const [currentInterviewerParagraph, setCurrentInterviewerParagraph] = useState("");
  
  // Pure interim (real-time, not is_final)
  const [currentCandidateInterim, setCurrentCandidateInterim] = useState("");
  const [currentInterviewerInterim, setCurrentInterviewerInterim] = useState("");

  // Refs
  const deepgramWsRef = useRef(null);
  const qaWsRef = useRef(null);
  const candidateStreamRef = useRef(null);
  const interviewerStreamRef = useRef(null);
  const candidateAudioContextRef = useRef(null);
  const interviewerAudioContextRef = useRef(null);
  const candidateProcessorRef = useRef(null);
  const interviewerProcessorRef = useRef(null);
  const transcriptEndRef = useRef(null);
  const copilotEndRef = useRef(null);

  // Pause detection timer refs
  const candidatePauseTimerRef = useRef(null);
  const interviewerPauseTimerRef = useRef(null);
  
  // Track state with refs to prevent race conditions
  const candidateParagraphRef = useRef("");
  const interviewerParagraphRef = useRef("");

  // ============================================================================
  // AUTH CHECK
  // ============================================================================
  useEffect(() => {
    if (!loading && !user) {
      navigate("/sign-in");
    }
  }, [user, loading, navigate]);

  // ============================================================================
  // AUTO-SCROLL
  // ============================================================================
  useEffect(() => {
    if (settings.autoScroll) {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [candidateTranscript, interviewerTranscript, currentCandidateParagraph, currentInterviewerParagraph, currentCandidateInterim, currentInterviewerInterim, settings.autoScroll]);

  useEffect(() => {
    if (settings.autoScroll) {
      copilotEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [qaList, currentAnswer, settings.autoScroll]);

  // ============================================================================
  // LEFT PANEL: DEEPGRAM DUAL-STREAM TRANSCRIPTION
  // ============================================================================

  const connectDeepgram = () => {
    return new Promise((resolve, reject) => {
      const languageMap = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Hindi": "hi",
        "Mandarin": "zh"
      };
      const language = languageMap[settings.audioLanguage] || "en";

      const ws = new WebSocket(`wss://interview-assist-1.onrender.com/ws/dual-transcribe?language=${language}`);

      ws.onopen = () => {
        console.log('✓ Deepgram connected');
        setDeepgramStatus("Connected");
        setTabAudioError("");
        resolve(ws);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'transcript') {
            handleDeepgramTranscript(data);
          } else if (data.type === 'error') {
            setTabAudioError(data.message);
          } else if (data.type === 'ready' || data.type === 'connected') {
            console.log('✓ Deepgram:', data.message);
          }
        } catch (e) {
          console.error('Deepgram parse error:', e);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ Deepgram error:', error);
        setTabAudioError('Connection error. Backend must be running.');
        setDeepgramStatus("Error");
        reject(error);
      };

      ws.onclose = () => {
        console.log('🔌 Deepgram closed');
        setDeepgramStatus("Disconnected");
      };

      deepgramWsRef.current = ws;
    });
  };

  const handleDeepgramTranscript = (data) => {
    const { stream, transcript, is_final, speech_final } = data;

    if (!transcript || !transcript.trim()) return;

    const pauseInterval = (settings.pauseInterval || 2.0) * 1000;

    if (stream === 'candidate') {
      if (candidatePauseTimerRef.current) {
        clearTimeout(candidatePauseTimerRef.current);
      }

      if (is_final || speech_final) {
        setCurrentCandidateInterim('');
        
        const newParagraph = candidateParagraphRef.current 
          ? `${candidateParagraphRef.current} ${transcript.trim()}` 
          : transcript.trim();
        candidateParagraphRef.current = newParagraph;
        
        setCurrentCandidateParagraph(newParagraph);

        candidatePauseTimerRef.current = setTimeout(() => {
          if (candidateParagraphRef.current) {
            const finalText = candidateParagraphRef.current;
            
            setCandidateTranscript(transcripts => [...transcripts, {
              text: finalText,
              timestamp: new Date().toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
              }),
              id: Date.now() + Math.random()
            }]);
            
            candidateParagraphRef.current = '';
            setCurrentCandidateParagraph('');
          }
        }, pauseInterval);
      } else {
        if (!candidateParagraphRef.current) {
          setCurrentCandidateInterim(transcript);
        }
      }
    } else if (stream === 'interviewer') {
      if (interviewerPauseTimerRef.current) {
        clearTimeout(interviewerPauseTimerRef.current);
      }

      if (is_final || speech_final) {
        setCurrentInterviewerInterim('');
        
        const newParagraph = interviewerParagraphRef.current 
          ? `${interviewerParagraphRef.current} ${transcript.trim()}` 
          : transcript.trim();
        interviewerParagraphRef.current = newParagraph;
        
        setCurrentInterviewerParagraph(newParagraph);

        interviewerPauseTimerRef.current = setTimeout(() => {
          if (interviewerParagraphRef.current) {
            const finalText = interviewerParagraphRef.current;
            
            setInterviewerTranscript(transcripts => [...transcripts, {
              text: finalText,
              timestamp: new Date().toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
              }),
              id: Date.now() + Math.random()
            }]);
            
            // ⭐ FIXED: Send final transcript to Q&A WebSocket
            if (qaWsRef.current && qaWsRef.current.readyState === WebSocket.OPEN) {
              qaWsRef.current.send(JSON.stringify({
                type: "transcript",
                transcript: finalText,
                is_final: true,
                speech_final: true
              }));
              console.log("📤 Sent to Q&A:", finalText.substring(0, 50) + "...");
            }
            
            interviewerParagraphRef.current = '';
            setCurrentInterviewerParagraph('');
          }
        }, pauseInterval);
      } else {
        if (!interviewerParagraphRef.current) {
          setCurrentInterviewerInterim(transcript);
        }
      }
    }
  };

  const startMicrophoneCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      candidateStreamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      candidateAudioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      candidateProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!deepgramWsRef.current || deepgramWsRef.current.readyState !== WebSocket.OPEN || isPaused) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(inputData.length);

        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        deepgramWsRef.current.send(JSON.stringify({
          type: 'candidate',
          audio: Array.from(pcm16)
        }));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      console.log('✓ Microphone started');
    } catch (err) {
      console.error('Microphone error:', err);
      setTabAudioError('Microphone denied');
      throw err;
    }
  };

  const startSystemAudioCapture = async () => {
    if (interviewerStreamRef.current) {
      console.warn('⚠️ Interviewer audio already capturing');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          width: { ideal: 1 },
          height: { ideal: 1 },
          frameRate: { ideal: 1 }
        },
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          suppressLocalAudioPlayback: false
        }
      });

      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.stop();
        stream.removeTrack(videoTrack);
      }

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        throw new Error('No audio track. Enable "Share audio".');
      }

      interviewerStreamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      interviewerAudioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      interviewerProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!deepgramWsRef.current || deepgramWsRef.current.readyState !== WebSocket.OPEN || isPaused) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(inputData.length);

        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        deepgramWsRef.current.send(JSON.stringify({
          type: 'interviewer',
          audio: Array.from(pcm16)
        }));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      console.log('✓ System audio started');
      setShowTabModal(false);
    } catch (err) {
      console.error('System audio error:', err);

      if (err.name === 'NotAllowedError') {
        setTabAudioError('Screen share denied');
      } else if (err.message.includes('No audio track')) {
        setTabAudioError('No audio. Check "Share tab audio"');
        setShowTabModal(true);
      } else {
        setTabAudioError(`Failed: ${err.message}`);
      }
      throw err;
    }
  };

  const stopDeepgramCapture = () => {
    if (candidatePauseTimerRef.current) {
      clearTimeout(candidatePauseTimerRef.current);
    }
    if (interviewerPauseTimerRef.current) {
      clearTimeout(interviewerPauseTimerRef.current);
    }

    if (candidateParagraphRef.current) {
      setCandidateTranscript(prev => [...prev, {
        text: candidateParagraphRef.current,
        timestamp: new Date().toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit"
        }),
        id: Date.now() + Math.random()
      }]);
      candidateParagraphRef.current = '';
      setCurrentCandidateParagraph('');
    }

    if (interviewerParagraphRef.current) {
      setInterviewerTranscript(prev => [...prev, {
        text: interviewerParagraphRef.current,
        timestamp: new Date().toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit"
        }),
        id: Date.now() + Math.random()
      }]);
      
      // Send final transcript to Q&A before clearing
      if (qaWsRef.current && qaWsRef.current.readyState === WebSocket.OPEN) {
        qaWsRef.current.send(JSON.stringify({
          type: "transcript",
          transcript: interviewerParagraphRef.current,
          is_final: true,
          speech_final: true
        }));
      }
      
      interviewerParagraphRef.current = '';
      setCurrentInterviewerParagraph('');
    }

    if (candidateProcessorRef.current) {
      candidateProcessorRef.current.disconnect();
      candidateProcessorRef.current = null;
    }
    if (candidateAudioContextRef.current) {
      candidateAudioContextRef.current.close();
      candidateAudioContextRef.current = null;
    }
    if (candidateStreamRef.current) {
      candidateStreamRef.current.getTracks().forEach(track => track.stop());
      candidateStreamRef.current = null;
    }

    if (interviewerProcessorRef.current) {
      interviewerProcessorRef.current.disconnect();
      interviewerProcessorRef.current = null;
    }
    if (interviewerAudioContextRef.current) {
      interviewerAudioContextRef.current.close();
      interviewerAudioContextRef.current = null;
    }
    if (interviewerStreamRef.current) {
      interviewerStreamRef.current.getTracks().forEach(track => track.stop());
      interviewerStreamRef.current = null;
    }

    if (deepgramWsRef.current) {
      deepgramWsRef.current.close();
      deepgramWsRef.current = null;
    }

    console.log('✓ Deepgram stopped');
  };

  // ============================================================================
  // RIGHT PANEL: Q&A WITH DEEPGRAM TRANSCRIPTS
  // ============================================================================

  const connectQA = () => {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket("wss://interview-assist-1.onrender.com/ws/live-interview");

      ws.onopen = () => {
        console.log("✅ Q&A connected");
        setQaStatus("Initializing...");

        const initMessage = {
          type: "init",
          domain: domain || "Technical",
          user_id: user?.id || null,
          persona_id: personaId || null,
          position: personaData?.position || "",
          company_name: personaData?.company_name || "",
          company_description: personaData?.company_description || "",
          job_description: personaData?.job_description || "",
          resume_text: personaData?.resume_text || "",
          resume_filename: personaData?.resume_filename || "",
          custom_style_prompt: settings.custom_style_prompt || null,
          settings: {
            audioLanguage: settings.audioLanguage || "English",
            pauseInterval: settings.pauseInterval || 2.0,
            advancedQuestionDetection: settings.advancedQuestionDetection !== false,
            selectedResponseStyleId: settings.selectedResponseStyleId || "concise",
            responseStyle: settings.responseStyle || "professional",
            defaultModel: settings.defaultModel || "gpt-4o-mini",
            programmingLanguage: settings.programmingLanguage || "Python",
            interviewInstructions: settings.interviewInstructions || "",
            messageDirection: settings.messageDirection || "bottom",
            autoScroll: settings.autoScroll !== false
          }
        };

        ws.send(JSON.stringify(initMessage));
        resolve(ws);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("📩 Q&A:", data.type, data);

          switch (data.type) {
            case "ready":
            case "connected":
              setQaStatus("🤖 Q&A Active");
              break;

            case "question_detected":
              if (data.question) {
                console.log("❓ Question:", data.question);
                setCurrentQuestion(data.question);
                setCurrentAnswer("");
                setIsGenerating(true);
                setIsStreamingComplete(false);
              }
              break;

            case "answer_ready":
              console.log("💬 Answer:", data);

              if (!data.answer || typeof data.answer !== 'string' || data.answer.trim() === '') {
                console.error("❌ Invalid answer:", data.answer);
                setIsGenerating(false);
                break;
              }

              console.log("✅ Valid answer:", data.answer.substring(0, 100) + "...");

              setCurrentAnswer(data.answer);
              setIsGenerating(false);

              const streamingDuration = data.answer.length * 20 + 500;

              setTimeout(() => {
                setIsStreamingComplete(true);

                addQA({
                  question: data.question || currentQuestion,
                  answer: data.answer
                });

                setTimeout(() => {
                  setCurrentQuestion("");
                  setCurrentAnswer("");
                  setIsStreamingComplete(false);
                }, 1000);
              }, streamingDuration);
              break;

            case "error":
              console.error("Q&A error:", data.message);
              setQaStatus(`⚠️ ${data.message}`);
              setIsGenerating(false);
              break;
          }
        } catch (err) {
          console.error("Parse error:", err);
        }
      };

      ws.onerror = (error) => {
        console.error("❌ Q&A error:", error);
        setQaStatus("Error");
        reject(error);
      };

      ws.onclose = () => {
        console.log("🔌 Q&A closed");
        setQaStatus("Disconnected");
      };

      qaWsRef.current = ws;
    });
  };

  const stopQA = () => {
    if (qaWsRef.current) {
      qaWsRef.current.close();
      qaWsRef.current = null;
    }
    console.log('✓ Q&A stopped');
  };

  const addQA = (qa) => {
    setQaList((prev) => {
      const isDuplicate = prev.some(
        (item) => item.question.trim().toLowerCase() === qa.question.trim().toLowerCase()
      );
      if (isDuplicate) return prev;

      const newQA = { ...qa, id: Date.now() + Math.random() };
      return [...prev, newQA];
    });
  };

  const startRecording = async () => {
    try {
      setTabAudioError("");

      await connectDeepgram();
      await startMicrophoneCapture();
      setShowTabModal(true);

      await connectQA();

      setIsRecording(true);
      setDeepgramStatus("🎤 Recording (Select Tab)");
      setQaStatus("🤖 Q&A Active");
    } catch (err) {
      console.error('Failed to start:', err);
      stopDeepgramCapture();
      stopQA();
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    stopDeepgramCapture();
    stopQA();
    setIsRecording(false);
    setDeepgramStatus("Stopped");
    setQaStatus("Stopped");
    
    candidateParagraphRef.current = '';
    interviewerParagraphRef.current = '';
    setCurrentCandidateParagraph('');
    setCurrentInterviewerParagraph('');
    setCurrentCandidateInterim('');
    setCurrentInterviewerInterim('');
  };

  const handleTabAudioSelection = async () => {
    try {
      await startSystemAudioCapture();
      setDeepgramStatus("🎤 Recording Active");
    } catch (err) {
      // Error handled in startSystemAudioCapture
    }
  };

  const generatePDF = () => {
    if (!qaList || qaList.length === 0) {
      alert("No Q&A data");
      return;
    }

    const doc = new jsPDF();
    let y = 10;

    doc.setFontSize(16);
    doc.text("Interview Q&A", 10, y);
    y += 10;

    if (personaData) {
      doc.setFontSize(10);
      doc.text(`${personaData.position} @ ${personaData.company_name}`, 10, y);
      y += 6;
    }

    if (domain) {
      doc.text(`Domain: ${domain}`, 10, y);
      y += 10;
    }

    doc.setFontSize(12);
    qaList.forEach((item, index) => {
      if (y > 260) {
        doc.addPage();
        y = 10;
      }

      doc.text(`Q${index + 1}: ${item.question}`, 10, y);
      y += 10;

      const lines = doc.splitTextToSize(`A: ${item.answer}`, 180);
      doc.text(lines, 10, y);
      y += lines.length * 7 + 5;
    });

    doc.save(`Interview_QnA_${new Date().toISOString().split('T')[0]}.pdf`);
  };

  useEffect(() => {
    return () => {
      stopDeepgramCapture();
      stopQA();
    };
  }, []);

  if (loading) {
    return (
      <div className="h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-white text-lg">Loading...</div>
      </div>
    );
  }

  if (!user) return null;

  const currentTranscript = activeView === "interviewer" ? interviewerTranscript : candidateTranscript;
  const currentParagraph = activeView === "interviewer" ? currentInterviewerParagraph : currentCandidateParagraph;
  const currentInterim = activeView === "interviewer" ? currentInterviewerInterim : currentCandidateInterim;

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/interview")}
              className="text-gray-400 hover:text-gray-200 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-xl font-semibold text-white">
                {personaData ? `${personaData.position} @ ${personaData.company_name}` : "Interview Assistant"}
              </h1>
              <p className="text-xs text-gray-500 mt-0.5">
                Real-time transcription & AI Q&A
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold text-lg transition-all transform hover:scale-105 shadow-2xl ${
                isRecording
                  ? "bg-red-600 hover:bg-red-700 shadow-red-500/50"
                  : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-blue-500/50"
              }`}
            >
              {isRecording ? (
                <>
                  <MicOff className="w-5 h-5" />
                  Stop Recording
                </>
              ) : (
                <>
                  <Mic className="w-5 h-5" />
                  Start Recording
                </>
              )}
            </button>

            {isRecording && (
              <div className="flex items-center gap-2 text-red-400 animate-pulse">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-sm font-medium">Recording</span>
              </div>
            )}

            <button
              onClick={generatePDF}
              className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
              disabled={qaList.length === 0}
            >
              <Download className="w-5 h-5" />
            </button>

            <button
              onClick={() => setShowSettings(true)}
              className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>

            <button
              onClick={() => navigate("/interview")}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors"
            >
              Exit
            </button>
          </div>
        </div>

        {personaData && (
          <div className="flex items-center gap-6 text-sm text-gray-400">
            <span><strong className="text-purple-400">Position:</strong> {personaData.position}</span>
            <span><strong className="text-purple-400">Company:</strong> {personaData.company_name}</span>
            {domain && <span><strong className="text-pink-400">Domain:</strong> {domain}</span>}
          </div>
        )}
      </header>

      {/* Tab Modal */}
      {showTabModal && isRecording && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 backdrop-blur-sm">
          <div className="bg-gray-900 rounded-xl p-6 w-full max-w-lg border border-gray-800 shadow-2xl">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold text-white">📡 Enable Interviewer Audio</h3>
              <button
                onClick={() => {
                  setShowTabModal(false);
                  if (!interviewerStreamRef.current) {
                    stopRecording();
                  }
                }}
                className="text-gray-400 hover:text-gray-200"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-yellow-200">
                    <p className="font-semibold mb-2">Enable "Share tab audio"</p>
                    <ol className="list-decimal list-inside space-y-1 text-xs">
                      <li>Click "Select Tab Audio"</li>
                      <li>Choose interview tab (Zoom/Meet)</li>
                      <li><strong className="text-yellow-300">Check "Share tab audio"</strong></li>
                      <li>Click "Share"</li>
                    </ol>
                  </div>
                </div>
              </div>

              {tabAudioError && (
                <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-3">
                  <p className="text-red-300 text-sm">{tabAudioError}</p>
                </div>
              )}

              <button
                onClick={handleTabAudioSelection}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-3 rounded-lg font-medium"
              >
                Select Tab Audio
              </button>

              <p className="text-xs text-gray-400 text-center">
                Microphone already active. This captures interviewer voice.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT: Deepgram Dual-Stream Transcription */}
        <div className="w-1/2 border-r border-gray-800 flex flex-col">
          <div className="bg-gray-900 border-b border-gray-800 flex items-center">
            <button
              onClick={() => setActiveView("interviewer")}
              className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${
                activeView === "interviewer"
                  ? "border-green-500 text-green-400 bg-gray-800"
                  : "border-transparent text-gray-400 hover:text-gray-300"
              }`}
            >
              <Volume2 className="w-4 h-4" />
              Interviewer
              {interviewerTranscript.length > 0 && (
                <span className="ml-1 text-xs bg-green-900/30 text-green-400 px-2 py-0.5 rounded-full">
                  {interviewerTranscript.length}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveView("candidate")}
              className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${
                activeView === "candidate"
                  ? "border-blue-500 text-blue-400 bg-gray-800"
                  : "border-transparent text-gray-400 hover:text-gray-300"
              }`}
            >
              <Mic className="w-4 h-4" />
              Candidate (You)
              {candidateTranscript.length > 0 && (
                <span className="ml-1 text-xs bg-blue-900/30 text-blue-400 px-2 py-0.5 rounded-full">
                  {candidateTranscript.length}
                </span>
              )}
            </button>

            <div className="ml-auto px-4 py-3">
              <button
                onClick={() => setIsPaused(!isPaused)}
                className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                disabled={!isRecording}
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            {currentTranscript.length === 0 && !currentParagraph && !currentInterim ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="mb-4">
                    {activeView === "interviewer" ? (
                      <Volume2 className="w-16 h-16 mx-auto text-gray-700" />
                    ) : (
                      <Mic className="w-16 h-16 mx-auto text-gray-700" />
                    )}
                  </div>
                  <p className="text-gray-500 mb-2 font-medium">
                    {activeView === "interviewer" ? "Interviewer speech" : "Your speech"}
                  </p>
                  <p className="text-gray-600 text-sm">Click "Start Recording" to begin</p>
                </div>
              </div>
            ) : (
              <>
                {currentTranscript.map((entry) => (
                  <div
                    key={entry.id}
                    className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span
                        className={`text-xs font-semibold px-2 py-1 rounded ${
                          activeView === "interviewer"
                            ? "text-green-400 bg-green-900/30"
                            : "text-blue-400 bg-blue-900/30"
                        }`}
                      >
                        {activeView === "interviewer" ? "Interviewer" : "You"}
                      </span>
                      <span className="text-xs text-gray-500">{entry.timestamp}</span>
                    </div>
                    <p className="text-gray-200 leading-relaxed">{entry.text}</p>
                  </div>
                ))}

                {currentParagraph && (
                  <div
                    className={`bg-gray-900/50 rounded-lg p-4 border-2 ${
                      activeView === "interviewer" ? "border-green-500/30" : "border-blue-500/30"
                    } animate-pulse`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span
                        className={`text-xs font-semibold flex items-center gap-2 px-2 py-1 rounded ${
                          activeView === "interviewer"
                            ? "text-green-300 bg-green-900/20"
                            : "text-blue-300 bg-blue-900/20"
                        }`}
                      >
                        {activeView === "interviewer" ? "Interviewer" : "You"}
                        <span className="text-xs">speaking...</span>
                      </span>
                    </div>
                    <p className="text-gray-200 leading-relaxed">{currentParagraph}</p>
                  </div>
                )}
                
                {!currentParagraph && currentInterim && (
                  <div
                    className={`bg-gray-900/50 rounded-lg p-4 border-2 ${
                      activeView === "interviewer" ? "border-green-500/20" : "border-blue-500/20"
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span
                        className={`text-xs font-semibold flex items-center gap-2 px-2 py-1 rounded ${
                          activeView === "interviewer"
                            ? "text-green-300 bg-green-900/20"
                            : "text-blue-300 bg-blue-900/20"
                        }`}
                      >
                        {activeView === "interviewer" ? "Interviewer" : "You"}
                        <span className="text-xs opacity-60">interim...</span>
                      </span>
                    </div>
                    <p className="text-gray-300 leading-relaxed opacity-70 italic">{currentInterim}</p>
                  </div>
                )}
              </>
            )}
            <div ref={transcriptEndRef} />
          </div>
        </div>

        {/* RIGHT: Q&A Copilot */}
        <div className="w-1/2 flex flex-col">
          <div className="bg-gray-900 px-6 py-3 border-b border-gray-800 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">Interview Copilot</h2>
              <p className="text-xs text-gray-500 mt-0.5">AI Q&A (Automatic)</p>
            </div>
            <div className="text-sm text-gray-400 flex items-center gap-2">
              <span className="bg-gray-800 px-3 py-1 rounded-full">
                {qaList.length} answer{qaList.length !== 1 ? 's' : ''}
              </span>
              {isGenerating && (
                <span className="flex items-center gap-1 text-purple-400">
                  <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                  generating...
                </span>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-4">
            {qaList.length === 0 && !currentQuestion ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="mb-4">
                    <div className="w-16 h-16 mx-auto bg-gray-800 rounded-full flex items-center justify-center">
                      <span className="text-3xl">🤖</span>
                    </div>
                  </div>
                  <p className="text-gray-500 mb-2 font-medium">AI answers appear here</p>
                  <p className="text-gray-600 text-sm">Click "Start Recording" to begin</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {qaList.length > 0 && <QAList qaList={qaList} />}

                {currentQuestion && (
                  <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 border-2 border-purple-500/50 rounded-lg p-5 shadow-lg">
                    <div className="mb-4">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xs font-semibold text-purple-300 bg-purple-900/50 px-3 py-1 rounded-full">
                          ❓ CURRENT QUESTION
                        </span>
                      </div>
                      <p className="text-white font-medium text-lg leading-relaxed">{currentQuestion}</p>
                    </div>

                    <div className="border-t border-purple-500/30 pt-4">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xs font-semibold text-green-300 bg-green-900/50 px-3 py-1 rounded-full">
                          💬 ANSWER
                        </span>
                        {isGenerating && !currentAnswer && (
                          <span className="text-xs text-purple-300 flex items-center gap-1">
                            <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                            Analyzing...
                          </span>
                        )}
                        {currentAnswer && !isStreamingComplete && (
                          <span className="text-xs text-green-400 flex items-center gap-1">
                            <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></span>
                            streaming...
                          </span>
                        )}
                      </div>

                      {currentAnswer ? (
                        <StreamingAnswer text={currentAnswer} isComplete={isStreamingComplete} />
                      ) : (
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                          <div
                            className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"
                            style={{ animationDelay: "0.4s" }}
                          ></div>
                          <span className="text-sm text-gray-300 ml-2">Preparing answer...</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
            <div ref={copilotEndRef} />
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="bg-gray-900 border-t border-gray-800 px-6 py-3 flex items-center justify-center gap-6">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              deepgramStatus.includes("Recording")
                ? "bg-green-500 animate-pulse"
                : deepgramStatus.includes("Error")
                ? "bg-red-500"
                : "bg-gray-600"
            }`}
          />
          <span className="text-sm text-gray-400">Deepgram: {deepgramStatus}</span>
        </div>

        <div className="h-4 w-px bg-gray-700" />

        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              qaStatus.includes("Active")
                ? "bg-blue-500 animate-pulse"
                : qaStatus.includes("Error")
                ? "bg-red-500"
                : "bg-gray-600"
            }`}
          />
          <span className="text-sm text-gray-400">Q&A: {qaStatus}</span>
        </div>

        {isGenerating && (
          <>
            <div className="h-4 w-px bg-gray-700" />
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
              <span className="text-sm text-purple-400">Processing...</span>
            </div>
          </>
        )}

        {isPaused && (
          <>
            <div className="h-4 w-px bg-gray-700" />
            <div className="flex items-center gap-2">
              <Pause className="w-3 h-3 text-orange-400" />
              <span className="text-sm text-orange-400">Paused</span>
            </div>
          </>
        )}
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 backdrop-blur-sm">
          <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md border border-gray-800 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold text-white">Settings</h3>
              <button onClick={() => setShowSettings(false)} className="text-gray-400 hover:text-gray-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <h4 className="text-sm font-medium mb-3 text-purple-400">Configuration</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Style:</span>
                    <span className="text-white font-medium">{settings.selectedResponseStyleId || settings.responseStyle}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Language:</span>
                    <span className="text-white font-medium">{settings.audioLanguage}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Pause Time:</span>
                    <span className="text-white font-medium">{settings.pauseInterval}s</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Coding:</span>
                    <span className="text-white font-medium">{settings.programmingLanguage}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Model:</span>
                    <span className="text-white font-medium">{settings.defaultModel}</span>
                  </div>
                </div>
              </div>

              <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4">
                <h4 className="text-sm font-medium mb-2 text-blue-300">💡 How It Works</h4>
                <ul className="text-gray-400 text-sm space-y-1 list-disc list-inside">
                  <li>Deepgram: Real-time transcription (dual-stream)</li>
                  <li>Q&A: Automatic answer generation from transcripts</li>
                  <li>Pause: {settings.pauseInterval}s creates new paragraph</li>
                  <li>Browser-based audio capture (100% client-side)</li>
                </ul>
              </div>

              <button
                onClick={() => setShowSettings(false)}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2.5 rounded-lg font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
