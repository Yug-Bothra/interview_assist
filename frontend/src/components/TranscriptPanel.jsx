import React, { useRef, useEffect } from "react";
import { Volume2, Mic, Pause, Play } from "lucide-react";

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
    <div className="flex flex-col h-full bg-[#111111]"> {/* lighter black */}
      
      {/* TOP TABS */}
      <div className="flex items-center justify-between border-b border-[#2A2A2A] bg-[#111111]">
        <div className="flex">
          {/* INTERVIEWER TAB */}
          <button
            onClick={() => onViewChange("interviewer")}
            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${
              activeView === "interviewer"
                ? "border-purple-500 text-purple-400 bg-[#181818]"   // updated highlight
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
          </button>

          {/* CANDIDATE TAB */}
          <button
            onClick={() => onViewChange("candidate")}
            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${
              activeView === "candidate"
                ? "border-amber-400 text-amber-300 bg-[#181818]" // fresh matching color
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
          </button>
        </div>

        {/* PAUSE BUTTON */}
        <button
          onClick={onPauseToggle}
          className="p-2 hover:bg-[#1A1A1A] rounded-lg transition-colors mr-2"
          disabled={!isRecording}
        >
          {isPaused ? (
            <Play className="w-4 h-4 text-gray-400" />
          ) : (
            <Pause className="w-4 h-4 text-gray-400" />
          )}
        </button>
      </div>

      {/* CONTENT AREA */}
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
            <p className="text-gray-500">Click "Start Recording" to begin</p>
          </div>
        ) : (
          <>
            {/* TRANSCRIPT BLOCKS */}
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

                <p className="text-gray-200 leading-relaxed">{entry.text}</p>

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

            {/* LIVE PARAGRAPH */}
            {currentParagraph && (
              <div className="bg-[#1A1A1A] border border-purple-400/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                  <span className="text-sm font-medium text-purple-300">
                    {activeView === "interviewer" ? "Interviewer" : "You"} speaking...
                  </span>
                </div>
                <p className="text-gray-200 leading-relaxed">{currentParagraph}</p>
              </div>
            )}

            {/* INTERIM */}
            {!currentParagraph && currentInterim && (
              <div className="bg-[#1A1A1A] border border-[#2A2A2A]/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 bg-gray-500 rounded-full" />
                  <span className="text-sm font-medium text-gray-400">
                    {activeView === "interviewer" ? "Interviewer" : "You"} interim...
                  </span>
                </div>
                <p className="text-gray-400 italic leading-relaxed">{currentInterim}</p>
              </div>
            )}

            <div ref={transcriptEndRef} />
          </>
        )}
      </div>
    </div>
  );
}
