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
  Headphones
} from "lucide-react";
import { jsPDF } from "jspdf";

// Streaming Text Component for word-by-word display
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

// Streaming Answer Component for AI responses
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

// QAList Component with numbering
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

export default function InterviewAssist() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, loading } = useAuth();

  // Data from navigation and localStorage
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

  // Settings from CopilotLaunchpad
  const [settings] = useState(() => {
    if (location.state?.settings) return location.state.settings;
    const saved = localStorage.getItem("copilotSettings");
    try {
      return saved
        ? JSON.parse(saved)
        : {
          responseStyle: "professional",
          audioLanguage: "English",
          pauseInterval: 0.4,
          advancedQuestionDetection: true,
          messageDirection: "bottom",
          autoScroll: true,
          programmingLanguage: "Python",
          audio_source_preference: "system"
        };
    } catch {
      return {
        responseStyle: "professional",
        audioLanguage: "English",
        pauseInterval: 0.4,
        advancedQuestionDetection: true,
        messageDirection: "bottom",
        autoScroll: true,
        programmingLanguage: "Python",
        audio_source_preference: "system"
      };
    }
  });

  // States
  const [qaList, setQaList] = useState([]);
  const [interviewerTranscript, setInterviewerTranscript] = useState([]);
  const [candidateTranscript, setCandidateTranscript] = useState([]);
  const [activeView, setActiveView] = useState("interviewer");
  const [interviewerAudioEnabled, setInterviewerAudioEnabled] = useState(true);
  const [candidateAudioEnabled, setCandidateAudioEnabled] = useState(false);
  const [interviewerStatus, setInterviewerStatus] = useState("Connecting...");
  const [candidateStatus, setCandidateStatus] = useState("Ready");
  const [showSettings, setShowSettings] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [audioSource, setAudioSource] = useState(settings.audio_source_preference || "system");
  const [screenShareWarning, setScreenShareWarning] = useState("");

  // Current Q&A states with streaming support
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStreamingComplete, setIsStreamingComplete] = useState(false);

  // Streaming states for transcripts
  const [streamingInterviewerText, setStreamingInterviewerText] = useState("");
  const [isInterviewerStreaming, setIsInterviewerStreaming] = useState(false);
  const [streamingCandidateText, setStreamingCandidateText] = useState("");
  const [isCandidateStreaming, setIsCandidateStreaming] = useState(false);

  // Refs
  const wsRef = useRef(null);
  const recognitionRef = useRef(null);
  const transcriptEndRef = useRef(null);
  const copilotEndRef = useRef(null);
  const retryRef = useRef(0);
  const recordingIntervalRef = useRef(null);
  const isConnectingRef = useRef(false);
  const cleanupRef = useRef(false);
  const maxRetries = 5;

  // Authentication check
  useEffect(() => {
    if (!loading && !user) {
      navigate("/sign-in");
    }
  }, [user, loading, navigate]);

  // Persist to localStorage
  useEffect(() => {
    if (personaId) localStorage.setItem("selectedPersona", personaId);
    if (personaData)
      localStorage.setItem("selectedPersonaData", JSON.stringify(personaData));
    if (domain) localStorage.setItem("selectedDomain", domain);
    if (settings) localStorage.setItem("copilotSettings", JSON.stringify(settings));
  }, [personaId, personaData, domain, settings]);

  // Auto-scroll for transcripts
  useEffect(() => {
    if (settings.autoScroll) {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [interviewerTranscript, candidateTranscript, streamingInterviewerText, streamingCandidateText, settings.autoScroll]);

  // Auto-scroll for copilot
  useEffect(() => {
    if (settings.autoScroll) {
      copilotEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [qaList, currentAnswer, settings.autoScroll]);

  // ============================================================================
  // BACKEND WEBSOCKET - HANDLES ONLY INTERVIEWER AUDIO
  // ============================================================================
  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    let pingInterval = null;
    let isManualDisconnect = false;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelays = [1000, 2000, 3000, 5000, 10000]; // Progressive backoff

    const cleanup = () => {
      isManualDisconnect = true;

      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
      }

      if (ws) {
        try {
          ws.close(1000, "Client cleanup");
        } catch (e) {
          console.log("WebSocket already closed:", e);
        }
        ws = null;
      }
    };

    const startPingInterval = () => {
      if (pingInterval) {
        clearInterval(pingInterval);
      }

      pingInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          try {
            ws.send(JSON.stringify({ type: "ping" }));
          } catch (e) {
            console.error("Ping failed:", e);
            if (!isManualDisconnect) {
              connectWebSocket();
            }
          }
        }
      }, 5000);
    };

    const connectWebSocket = () => {
      if (!interviewerAudioEnabled || isManualDisconnect) {
        setInterviewerStatus("Disabled");
        return;
      }

      if (ws) {
        try {
          ws.close();
        } catch (e) {
          console.log("Error closing existing connection:", e);
        }
        ws = null;
      }

      if (reconnectAttempts >= maxReconnectAttempts) {
        setInterviewerStatus("Connection failed ❌ (Max retries)");
        console.error("Max reconnection attempts reached");
        return;
      }

      setInterviewerStatus(
        reconnectAttempts > 0
          ? `Reconnecting... (${reconnectAttempts}/${maxReconnectAttempts})`
          : "Connecting..."
      );

      try {
        ws = new WebSocket("ws://127.0.0.1:8000/ws/live-interview");

        const connectionTimeout = setTimeout(() => {
          if (ws && ws.readyState !== WebSocket.OPEN) {
            console.error("Connection timeout");
            ws.close();
          }
        }, 5000);

        ws.onopen = () => {
          clearTimeout(connectionTimeout);
          reconnectAttempts = 0;
          setInterviewerStatus("Initializing...");
          console.log("✅ WebSocket connected");

          const initMessage = {
            type: "init",
            domain: domain || "Technical",
            user_id: user?.id || null,
            persona_id: personaId || null,
            position: personaData?.position || "",
            company_name: personaData?.company_name || "",
            settings: {
              audioLanguage: settings.audioLanguage || "en",
              pauseInterval: settings.pauseInterval || 0.4,
              advancedQuestionDetection: settings.advancedQuestionDetection !== false,
              selectedResponseStyleId: settings.selectedResponseStyleId || "concise",
              interviewInstructions: settings.interviewInstructions || "",
              programmingLanguage: settings.programmingLanguage || "Python"
            }
          };

          try {
            ws.send(JSON.stringify(initMessage));
            console.log("✓ Initialization message sent");
            startPingInterval();
          } catch (e) {
            console.error("Error sending init message:", e);
          }
        };

        ws.onmessage = async (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log("📨 Received:", data.type);

            switch (data.type) {
              case "ready":
              case "connected":
                const sourceLabel = audioSource === "tab" ? "Tab Audio" : "System Audio";
                setInterviewerStatus(`🎤 ${sourceLabel} Active`);
                break;

              case "transcript":
                if (data.text) {
                  addInterviewerTranscript(data.text);
                }
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
                if (data.answer) {
                  console.log("✓ Answer received");
                  setCurrentAnswer(data.answer);
                  setIsGenerating(false);

                  setTimeout(() => {
                    setIsStreamingComplete(true);
                    addQA({
                      question: data.question || currentQuestion,
                      answer: data.answer
                    });

                    setTimeout(() => {
                      setCurrentQuestion("");
                      setCurrentAnswer("");
                    }, 2000);
                  }, data.answer.length * 20 + 500);
                }
                break;

              case "pong":
                console.log("💓 Keepalive");
                break;

              case "error":
                console.error("Backend error:", data.message);
                setInterviewerStatus(`⚠️ ${data.message || "Error"}`);
                setIsGenerating(false);
                break;

              default:
                console.log("Unknown message type:", data.type);
            }
          } catch (err) {
            console.error("Message parse error:", err);
          }
        };

        ws.onerror = (error) => {
          console.error("❌ WebSocket error:", error);
          clearTimeout(connectionTimeout);
        };

        ws.onclose = (event) => {
          clearTimeout(connectionTimeout);

          if (pingInterval) {
            clearInterval(pingInterval);
            pingInterval = null;
          }

          console.log(`🔌 WebSocket closed: ${event.code} - ${event.reason || "No reason"}`);

          if (isManualDisconnect || !interviewerAudioEnabled) {
            setInterviewerStatus("Disconnected");
            return;
          }

          const shouldReconnect =
            event.code !== 1000 &&
            event.code !== 1001 &&
            reconnectAttempts < maxReconnectAttempts;

          if (shouldReconnect) {
            reconnectAttempts++;
            const delay = reconnectDelays[Math.min(reconnectAttempts - 1, reconnectDelays.length - 1)];

            setInterviewerStatus(
              `Reconnecting in ${delay / 1000}s... (${reconnectAttempts}/${maxReconnectAttempts})`
            );

            reconnectTimer = setTimeout(() => {
              console.log(`🔄 Attempting reconnection ${reconnectAttempts}/${maxReconnectAttempts}`);
              connectWebSocket();
            }, delay);
          } else {
            setInterviewerStatus("Connection closed ❌");
          }
        };

      } catch (error) {
        console.error("Failed to create WebSocket:", error);
        setInterviewerStatus("Connection failed ❌");

        if (reconnectAttempts < maxReconnectAttempts && !isManualDisconnect) {
          reconnectAttempts++;
          const delay = reconnectDelays[Math.min(reconnectAttempts - 1, reconnectDelays.length - 1)];
          reconnectTimer = setTimeout(connectWebSocket, delay);
        }
      }
    };

    if (interviewerAudioEnabled) {
      connectWebSocket();
    } else {
      cleanup();
    }

    return () => {
      cleanup();
    };
  }, [
    interviewerAudioEnabled,
    personaId,
    domain,
    settings,
    audioSource,
    user,
    personaData
  ]);

  // ============================================================================
  // FRONTEND - CANDIDATE MICROPHONE (WEB SPEECH API) - RUNS INDEPENDENTLY
  // ============================================================================
  useEffect(() => {
    if (!candidateAudioEnabled) {
      if (recordingIntervalRef.current) {
        clearTimeout(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {
          console.log("Recognition already stopped:", e);
        }
        recognitionRef.current = null;
      }
      setCandidateStatus("Disabled");
      return;
    }

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      setCandidateStatus("Speech Recognition Not Supported ❌");
      return;
    }

    let isCleaningUp = false;
    let currentText = '';

    const startSpeechRecognition = () => {
      try {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = settings.audioLanguage === 'English' ? 'en-US' : 'en-US';
        recognition.maxAlternatives = 1;

        recognitionRef.current = recognition;

        recognition.onstart = () => {
          if (isCleaningUp) return;
          setCandidateStatus("🎤 Microphone Active");
          console.log("✓ Candidate microphone started (Web Speech API)");
        };

        recognition.onresult = (event) => {
          if (isCleaningUp || isPaused) return;

          let interimTranscript = '';
          let finalTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript;
            } else {
              interimTranscript += transcript;
            }
          }

          if (finalTranscript) {
            currentText += (currentText ? ' ' : '') + finalTranscript;

            if (recordingIntervalRef.current) {
              clearTimeout(recordingIntervalRef.current);
            }

            const pauseTime = (settings.pauseInterval || 0.4) * 1000;
            recordingIntervalRef.current = setTimeout(() => {
              if (currentText.trim() && !isCleaningUp) {
                addCandidateTranscript(currentText.trim());
                currentText = '';
              }
            }, pauseTime);
          }
        };

        recognition.onerror = (event) => {
          if (isCleaningUp) return;

          console.error('Speech recognition error:', event.error);

          if (event.error === 'no-speech') {
            if (candidateAudioEnabled && !isCleaningUp) {
              setTimeout(() => {
                if (recognitionRef.current && candidateAudioEnabled) {
                  try {
                    recognition.start();
                  } catch (e) {
                    console.log("Already started:", e);
                  }
                }
              }, 100);
            }
          } else if (event.error === 'aborted') {
            return;
          } else {
            setCandidateStatus(`⚠️ Error: ${event.error}`);
          }
        };

        recognition.onend = () => {
          if (isCleaningUp) return;

          if (currentText.trim()) {
            addCandidateTranscript(currentText.trim());
            currentText = '';
          }

          if (candidateAudioEnabled && !isCleaningUp) {
            setTimeout(() => {
              if (recognitionRef.current && candidateAudioEnabled) {
                try {
                  recognition.start();
                } catch (e) {
                  console.log("Already started:", e);
                }
              }
            }, 100);
          }
        };

        recognition.start();
      } catch (err) {
        console.error("Speech recognition error:", err);
        if (!isCleaningUp) {
          setCandidateStatus("Microphone Access Denied ❌");
        }
      }
    };

    startSpeechRecognition();

    return () => {
      isCleaningUp = true;

      if (recordingIntervalRef.current) {
        clearTimeout(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }

      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {
          console.log("Error stopping recognition:", e);
        }
        recognitionRef.current = null;
      }
    };
  }, [candidateAudioEnabled, isPaused, settings.pauseInterval, settings.audioLanguage]);

  const addInterviewerTranscript = (text) => {
    const timestamp = new Date().toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit"
    });

    setStreamingInterviewerText(text);
    setIsInterviewerStreaming(true);

    setTimeout(() => {
      setInterviewerTranscript((prev) => {
        const newTranscript = { text, timestamp, id: Date.now() + Math.random() };
        return [...prev, newTranscript];
      });
      setIsInterviewerStreaming(false);
      setStreamingInterviewerText("");
    }, text.split(' ').length * 80 + 200);
  };

  const addCandidateTranscript = (text) => {
    const timestamp = new Date().toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit"
    });

    setStreamingCandidateText(text);
    setIsCandidateStreaming(true);

    setTimeout(() => {
      setCandidateTranscript((prev) => {
        const newTranscript = { text, timestamp, id: Date.now() + Math.random() };
        return [...prev, newTranscript];
      });
      setIsCandidateStreaming(false);
      setStreamingCandidateText("");
    }, text.split(' ').length * 80 + 200);
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

  const generatePDF = () => {
    if (!qaList || qaList.length === 0) {
      alert("No Q&A data to export");
      return;
    }

    const doc = new jsPDF();
    let y = 10;

    doc.setFontSize(16);
    doc.text("Interview Q&A Transcript", 10, y);
    y += 10;

    if (personaData) {
      doc.setFontSize(10);
      doc.text(`Position: ${personaData.position} @ ${personaData.company_name}`, 10, y);
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

      const questionNum = index + 1;
      doc.text(`Q${questionNum}: ${item.question}`, 10, y);
      y += 10;

      const lines = doc.splitTextToSize(`A: ${item.answer}`, 180);
      doc.text(lines, 10, y);
      y += lines.length * 7 + 5;
    });

    const filename = `Interview_QnA_${new Date().toISOString().split('T')[0]}.pdf`;
    doc.save(filename);
  };

  if (loading) {
    return (
      <div className="h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-white text-lg">Loading...</div>
      </div>
    );
  }

  if (!user) return null;

  const currentTranscript = activeView === "interviewer" ? interviewerTranscript : candidateTranscript;
  const isCurrentlyStreaming = activeView === "interviewer" ? isInterviewerStreaming : isCandidateStreaming;
  const currentStreamingText = activeView === "interviewer" ? streamingInterviewerText : streamingCandidateText;

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        {screenShareWarning && (
          <div className="mb-3 bg-yellow-900/50 border border-yellow-600/50 rounded-lg px-4 py-3 flex items-center justify-between animate-pulse">
            <span className="text-yellow-200 text-sm flex items-center gap-2">
              <span className="text-lg">⚠️</span>
              {screenShareWarning}
            </span>
            <button
              onClick={() => setScreenShareWarning("")}
              className="text-yellow-400 hover:text-yellow-200 transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/interview")}
              className="text-gray-400 hover:text-gray-200 transition-colors"
              title="Back to Interview Dashboard"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-xl font-semibold text-white">
                {personaData
                  ? `${personaData.position} @ ${personaData.company_name}`
                  : "Interview Assistant"}
              </h1>
              <p className="text-xs text-gray-500 mt-0.5">Backend: Interviewer | Frontend: Candidate</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                setAudioSource(audioSource === "tab" ? "system" : "tab");
              }}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${audioSource === "tab"
                  ? "bg-purple-600 text-white shadow-lg shadow-purple-500/30"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                }`}
              title={audioSource === "tab" ? "Using Tab Audio" : "Using System Audio"}
            >
              <Headphones className="w-4 h-4" />
              {audioSource === "tab" ? "Tab Audio" : "System Audio"}
            </button>

            <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-1">
              <button
                onClick={() => setInterviewerAudioEnabled(!interviewerAudioEnabled)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${interviewerAudioEnabled
                    ? "bg-green-600 text-white shadow-lg shadow-green-500/30"
                    : "bg-gray-700 text-gray-400 hover:bg-gray-650"
                  }`}
                title="Backend: Stereo Mix/Tab Audio"
              >
                Interviewer (Backend)
              </button>

              <button
                onClick={() => setCandidateAudioEnabled(!candidateAudioEnabled)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${candidateAudioEnabled
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-500/30"
                    : "bg-gray-700 text-gray-400 hover:bg-gray-650"
                  }`}
                title="Frontend: Browser Microphone"
              >
                <Mic className="w-4 h-4" />
                Candidate (Frontend)
              </button>
            </div>

            <button
              onClick={generatePDF}
              className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Export PDF"
              disabled={qaList.length === 0}
            >
              <Download className="w-5 h-5" />
            </button>

            <button
              onClick={() => setShowSettings(true)}
              className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
              title="View Settings"
            >
              <Settings className="w-5 h-5" />
            </button>

            <button
              onClick={() => navigate("/interview")}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors shadow-lg shadow-red-500/20"
              title="Exit Interview"
            >
              Exit
            </button>
          </div>
        </div>

        {personaData && (
          <div className="flex items-center gap-6 text-sm text-gray-400">
            <span>
              <strong className="text-purple-400">Position:</strong> {personaData.position}
            </span>
            <span>
              <strong className="text-purple-400">Company:</strong> {personaData.company_name}
            </span>
            {domain && (
              <span>
                <strong className="text-pink-400">Domain:</strong> {domain}
              </span>
            )}
          </div>
        )}
      </header>

      {/* Main Content - Split Panel */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Transcription */}
        <div className="w-1/2 border-r border-gray-800 flex flex-col">
          <div className="bg-gray-900 border-b border-gray-800 flex items-center">
            <button
              onClick={() => setActiveView("interviewer")}
              className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${activeView === "interviewer"
                  ? "border-green-500 text-green-400 bg-gray-800"
                  : "border-transparent text-gray-400 hover:text-gray-300 hover:bg-gray-850"
                }`}
            >
              <Volume2 className="w-4 h-4" />
              Interviewer (Backend)
              {interviewerTranscript.length > 0 && (
                <span className="ml-1 text-xs bg-green-900/30 text-green-400 px-2 py-0.5 rounded-full">
                  {interviewerTranscript.length}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveView("candidate")}
              className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${activeView === "candidate"
                  ? "border-blue-500 text-blue-400 bg-gray-800"
                  : "border-transparent text-gray-400 hover:text-gray-300 hover:bg-gray-850"
                }`}
            >
              <Mic className="w-4 h-4" />
              Candidate (Frontend)
              {candidateTranscript.length > 0 && (
                <span className="ml-1 text-xs bg-blue-900/30 text-blue-400 px-2 py-0.5 rounded-full">
                  {candidateTranscript.length}
                </span>
              )}
            </button>

            <div className="ml-auto px-4 py-3 flex items-center gap-2">
              <button
                onClick={() => setIsPaused(!isPaused)}
                className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                title={isPaused ? "Resume transcription" : "Pause transcription"}
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Transcription Content */}
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            {currentTranscript.length === 0 && !isCurrentlyStreaming ? (
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
                    {activeView === "interviewer"
                      ? "Interviewer speech will appear here (Backend)"
                      : "Your responses will appear here (Frontend)"}
                  </p>
                  <p className="text-gray-600 text-sm">
                    {activeView === "interviewer"
                      ? "Enable interviewer audio above"
                      : "Enable your microphone above"}
                  </p>
                </div>
              </div>
            ) : (
              <>
                {currentTranscript.map((entry) => (
                  <div key={entry.id} className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors">
                    <div className="flex justify-between items-start mb-2">
                      <span
                        className={`text-xs font-semibold px-2 py-1 rounded ${activeView === "interviewer"
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

                {isCurrentlyStreaming && currentStreamingText && (
                  <div className={`bg-gray-900/70 rounded-lg p-4 border-2 ${activeView === "interviewer" ? "border-green-500/50" : "border-blue-500/50"
                    } shadow-lg ${activeView === "interviewer" ? "shadow-green-500/20" : "shadow-blue-500/20"
                    }`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className={`text-xs font-semibold flex items-center gap-2 px-2 py-1 rounded ${activeView === "interviewer"
                          ? "text-green-400 bg-green-900/30"
                          : "text-blue-400 bg-blue-900/30"
                        }`}>
                        {activeView === "interviewer" ? "Interviewer" : "You"}
                        <span className="inline-flex items-center gap-1">
                          <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${activeView === "interviewer" ? "bg-green-400" : "bg-blue-400"
                            }`}></span>
                          <span className="text-xs">speaking...</span>
                        </span>
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date().toLocaleTimeString("en-US", {
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit"
                        })}
                      </span>
                    </div>
                    <StreamingText
                      text={currentStreamingText}
                      isComplete={false}
                      className="text-gray-200"
                    />
                  </div>
                )}
              </>
            )}
            <div ref={transcriptEndRef} />
          </div>
        </div>

        {/* Right Panel - Interview Copilot */}
        <div className="w-1/2 flex flex-col">
          <div className="bg-gray-900 px-6 py-3 border-b border-gray-800 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">Interview Copilot</h2>
              <p className="text-xs text-gray-500 mt-0.5">AI-powered answer generation</p>
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
            {(currentQuestion || isGenerating) && (
              <div className="mb-6 bg-gradient-to-r from-purple-900/30 to-blue-900/30 border-2 border-purple-500/50 rounded-lg p-4 shadow-lg shadow-purple-500/20 animate-pulse">
                <div className="mb-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-semibold text-purple-300 bg-purple-900/50 px-3 py-1 rounded-full">
                      ❓ CURRENT QUESTION
                    </span>
                    {isGenerating && (
                      <span className="inline-flex items-center gap-1 text-xs text-purple-300">
                        <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                        Generating answer...
                      </span>
                    )}
                  </div>
                  <p className="text-white font-medium text-lg leading-relaxed">{currentQuestion}</p>
                </div>

                {currentAnswer && (
                  <div className="border-t border-purple-500/30 pt-3 mt-3">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs font-semibold text-green-300 bg-green-900/50 px-3 py-1 rounded-full">💬 ANSWER</span>
                      {!isStreamingComplete && (
                        <span className="text-xs text-green-400 flex items-center gap-1">
                          <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></span>
                          streaming...
                        </span>
                      )}
                    </div>
                    <StreamingAnswer text={currentAnswer} isComplete={isStreamingComplete} />
                  </div>
                )}

                {isGenerating && !currentAnswer && (
                  <div className="border-t border-purple-500/30 pt-3 mt-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                      <span className="text-sm text-gray-300 ml-2">Analyzing question and generating response...</span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {qaList.length === 0 && !currentQuestion ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="mb-4">
                    <div className="w-16 h-16 mx-auto bg-gray-800 rounded-full flex items-center justify-center">
                      <span className="text-3xl">🤖</span>
                    </div>
                  </div>
                  <p className="text-gray-500 mb-2 font-medium">AI answers will appear here</p>
                  <p className="text-gray-600 text-sm">Enable interviewer audio to get started</p>
                </div>
              </div>
            ) : (
              <>
                {qaList.length > 0 && (
                  <div className="mb-4">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider bg-gray-900 px-3 py-2 rounded-lg inline-block">
                      📚 Previous Answers ({qaList.length})
                    </h3>
                  </div>
                )}
                <QAList qaList={qaList} />
              </>
            )}
            <div ref={copilotEndRef} />
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="bg-gray-900 border-t border-gray-800 px-6 py-3 flex items-center justify-center gap-6">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${interviewerStatus.includes("Active")
                ? "bg-green-500 animate-pulse shadow-lg shadow-green-500/50"
                : interviewerStatus.includes("Error") || interviewerStatus.includes("failed")
                  ? "bg-red-500 shadow-lg shadow-red-500/50"
                  : interviewerStatus === "Disabled"
                    ? "bg-gray-600"
                    : "bg-yellow-500 animate-pulse"
              }`}
          />
          <span className="text-sm text-gray-400">Backend: {interviewerStatus}</span>
        </div>

        <div className="h-4 w-px bg-gray-700" />

        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${candidateStatus.includes("Active")
                ? "bg-blue-500 animate-pulse shadow-lg shadow-blue-500/50"
                : candidateStatus.includes("Denied") || candidateStatus.includes("Not Supported")
                  ? "bg-red-500 shadow-lg shadow-red-500/50"
                  : candidateStatus === "Disabled"
                    ? "bg-gray-600"
                    : "bg-yellow-500 animate-pulse"
              }`}
          />
          <span className="text-sm text-gray-400">Frontend: {candidateStatus}</span>
        </div>

        {isGenerating && (
          <>
            <div className="h-4 w-px bg-gray-700" />
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse shadow-lg shadow-purple-500/50" />
              <span className="text-sm text-purple-400">AI Processing...</span>
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
          <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md border border-gray-800 max-h-[90vh] overflow-y-auto shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold text-white">Active Settings</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="text-gray-400 hover:text-gray-200 transition-colors"
                title="Close settings"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <h4 className="text-sm font-medium mb-3 text-purple-400">Current Configuration</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Response Style:</span>
                    <span className="text-white font-medium capitalize">{settings.responseStyle}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Audio Language:</span>
                    <span className="text-white font-medium">{settings.audioLanguage}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Pause Interval:</span>
                    <span className="text-white font-medium">{settings.pauseInterval}s</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Question Detection:</span>
                    <span className="text-white font-medium">
                      {settings.advancedQuestionDetection ? "Advanced" : "Standard"}
                    </span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Auto Scroll:</span>
                    <span className="text-white font-medium">{settings.autoScroll ? "Enabled" : "Disabled"}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Programming Language:</span>
                    <span className="text-white font-medium">{settings.programmingLanguage}</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <h4 className="text-sm font-medium mb-3 text-green-400">Active Audio Sources</h4>
                <div className="space-y-2 text-sm text-gray-400">
                  <div className="flex items-center justify-between py-1">
                    <span>Backend: Interviewer ({audioSource === "tab" ? "Tab" : "System"})</span>
                    <span className={`font-medium ${interviewerAudioEnabled ? "text-green-400" : "text-gray-500"}`}>
                      {interviewerAudioEnabled ? "✓ Enabled" : "✗ Disabled"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-1">
                    <span>Frontend: Candidate (Web Speech API)</span>
                    <span className={`font-medium ${candidateAudioEnabled ? "text-blue-400" : "text-gray-500"}`}>
                      {candidateAudioEnabled ? "✓ Enabled" : "✗ Disabled"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4">
                <h4 className="text-sm font-medium mb-2 text-blue-300 flex items-center gap-2">
                  <span>💡</span>
                  Architecture Notes
                </h4>
                <ul className="text-gray-400 text-sm space-y-2 list-disc list-inside">
                  <li><strong className="text-blue-400">Backend (Python):</strong> Handles interviewer audio via Stereo Mix/Tab capture + Whisper transcription</li>
                  <li><strong className="text-green-400">Frontend (React):</strong> Handles candidate audio via browser's Web Speech API</li>
                  <li><strong className="text-purple-400">Parallel Processing:</strong> Both audio sources work independently and simultaneously</li>
                  <li>Tab audio requires selecting a specific browser tab/window (not entire screen)</li>
                  <li>To modify settings, return to Copilot Launchpad before starting</li>
                </ul>
              </div>

              <button
                onClick={() => setShowSettings(false)}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2.5 rounded-lg font-medium transition-colors shadow-lg shadow-purple-500/20"
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