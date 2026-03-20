import React, { useRef, useEffect } from "react";
import { Volume2, Mic, Pause, Play, Activity } from "lucide-react";

export default function TranscriptPanel({
  activeView,
  onViewChange,
  interviewerTranscript,
  candidateTranscript,
  currentInterviewerParagraph,
  currentCandidateParagraph,
  currentInterviewerInterim,
  currentCandidateInterim,
  isPaused,
  onPauseToggle,
  isRecording,
  onManualGenerate,
  autoScroll,
}) {
  const transcriptEndRef = useRef(null);

  useEffect(() => {
    if (autoScroll) {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [
    candidateTranscript,
    interviewerTranscript,
    currentCandidateParagraph,
    currentInterviewerParagraph,
    currentCandidateInterim,
    currentInterviewerInterim,
    autoScroll,
  ]);

  const currentTranscript =
    activeView === "interviewer" ? interviewerTranscript : candidateTranscript;
  const currentParagraph =
    activeView === "interviewer"
      ? currentInterviewerParagraph
      : currentCandidateParagraph;
  const currentInterim =
    activeView === "interviewer"
      ? currentInterviewerInterim
      : currentCandidateInterim;

  return (
    <div className="flex flex-col h-full bg-[#111111]">
      
      {/* TOP TABS */}
      <div className="flex items-center justify-between border-b border-[#2A2A2A] bg-[#111111]">
        <div className="flex">
          {/* INTERVIEWER TAB */}
          <button
            onClick={() => onViewChange("interviewer")}
            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${
              activeView === "interviewer"
                ? "border-purple-500 text-purple-400 bg-[#181818]"
                : "border-transparent text-gray-400 hover:text-gray-300"
            }`}
          >
            <Volume2 className="w-4 h-4" />
            Interviewer
            {interviewerTranscript.length > 0 && (
              <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded-full">
                {interviewerTranscript.length}
              </span>
            )}
            {currentInterviewerInterim && (
              <Activity className="w-3 h-3 text-purple-400 animate-pulse" />
            )}
          </button>

          {/* CANDIDATE TAB */}
          <button
            onClick={() => onViewChange("candidate")}
            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${
              activeView === "candidate"
                ? "border-amber-400 text-amber-300 bg-[#181818]"
                : "border-transparent text-gray-400 hover:text-gray-300"
            }`}
          >
            <Mic className="w-4 h-4" />
            Candidate (You)
            {candidateTranscript.length > 0 && (
              <span className="px-2 py-0.5 bg-amber-400/20 text-amber-300 text-xs rounded-full">
                {candidateTranscript.length}
              </span>
            )}
            {currentCandidateInterim && (
              <Activity className="w-3 h-3 text-amber-400 animate-pulse" />
            )}
          </button>
        </div>

        {/* PAUSE BUTTON */}
        <button
          onClick={onPauseToggle}
          className="p-2 hover:bg-[#1A1A1A] rounded-lg transition-colors mr-2"
          disabled={!isRecording}
          title={isPaused ? "Resume" : "Pause"}
        >
          {isPaused ? (
            <Play className="w-4 h-4 text-gray-400" />
          ) : (
            <Pause className="w-4 h-4 text-gray-400" />
          )}
        </button>
      </div>

      {/* CONTENT AREA - FIXED HEIGHT, SCROLLABLE */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {currentTranscript.length === 0 && !currentParagraph && !currentInterim ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-20 h-20 bg-[#181818] rounded-full flex items-center justify-center mb-4 border border-[#2A2A2A]">
              {activeView === "interviewer" ? (
                <Volume2 className="w-10 h-10 text-gray-600" />
              ) : (
                <Mic className="w-10 h-10 text-gray-600" />
              )}
            </div>
            <h3 className="text-xl font-semibold text-gray-400 mb-2">
              {activeView === "interviewer" ? "Interviewer speech" : "Your speech"}
            </h3>
            <p className="text-gray-500">Waiting for speech...</p>
          </div>
        ) : (
          <>
            {/* FINALIZED TRANSCRIPT BLOCKS */}
            {currentTranscript.map((entry) => (
              <div
                key={entry.id}
                className="bg-[#181818] rounded-lg p-4 border border-[#2A2A2A]"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-purple-400">
                    {entry.stream === "interviewer" ? "Interviewer" : "You"}
                  </span>
                  <span className="text-xs text-gray-500">{entry.timestamp}</span>
                </div>

                <p className="text-gray-200 leading-relaxed whitespace-pre-wrap">{entry.text}</p>

                {entry.stream === "interviewer" && (
                  <button
                    onClick={() => onManualGenerate(entry.text)}
                    className="mt-3 px-3 py-1 border border-[#2A2A2A] bg-[#1A1A1A] text-gray-300 rounded text-sm hover:bg-[#222222] transition-colors"
                  >
                    Generate Answer
                  </button>
                )}
              </div>
            ))}

            {/* LIVE PARAGRAPH (FINAL TEXT ACCUMULATING) */}
            {currentParagraph && (
              <div className="bg-[#1A1A1A] border-2 border-purple-400/40 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                  <span className="text-sm font-medium text-purple-300">
                    {activeView === "interviewer" ? "Interviewer" : "You"} speaking...
                  </span>
                  <span className="text-xs text-gray-500">(final text)</span>
                </div>
                <p className="text-gray-200 leading-relaxed whitespace-pre-wrap">
                  {currentParagraph}
                </p>
              </div>
            )}

            {/* LIVE INTERIM (REAL-TIME STREAMING FROM DEEPGRAM) */}
            {currentInterim && (
              <div className="bg-[#1A1A1A] border-2 border-dashed border-yellow-500/40 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-3 h-3 text-yellow-400 animate-pulse" />
                  <span className="text-sm font-medium text-yellow-400">
                    Live interim
                  </span>
                  <span className="text-xs text-gray-500">(updating)</span>
                </div>
                <p className="text-gray-300 leading-relaxed whitespace-pre-wrap font-mono text-sm">
                  {currentInterim}
                </p>
              </div>
            )}

            <div ref={transcriptEndRef} />
          </>
        )}
      </div>
    </div>
  );
}