import React, { useRef, useEffect } from "react";
import MarkdownRenderer from "./MarkdownRenderer";
import { Sparkles, MessageCircle } from "lucide-react";

function StreamingAnswer({ text, isComplete }) {
  const [displayedText, setDisplayedText] = React.useState("");
  const [currentIndex, setCurrentIndex] = React.useState(0);

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
    <div className="relative">
      <MarkdownRenderer content={displayedText} />
      {!isComplete && currentIndex < text.length && (
        <span className="inline-block w-1 h-4 bg-purple-400 animate-pulse ml-1 rounded-sm" />
      )}
    </div>
  );
}

function QAList({ qaList }) {
  return (
    <div className="space-y-5">
      {qaList.map((item, index) => (
        <div
          key={item.id}
          className="bg-[#181818] rounded-xl border border-[#262626] p-5 transition-all hover:border-[#333]"
        >
          {/* QUESTION HEADER */}
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 bg-[#222] rounded-md flex items-center justify-center">
              <span className="text-purple-400 text-sm">❓</span>
            </div>
            <span className="text-sm font-bold text-purple-300">
              QUESTION #{index + 1}
            </span>
          </div>

          {/* QUESTION TEXT */}
          <p className="text-gray-200 mb-4 leading-relaxed">{item.question}</p>

          {/* ANSWER BLOCK */}
          <div className="bg-[#1f1f1f] border border-[#2a2a2a] rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <MessageCircle className="w-4 h-4 text-purple-300" />
              <span className="text-sm font-bold text-purple-300">AI ANSWER</span>
            </div>

            <div className="text-gray-200 leading-relaxed">
              <MarkdownRenderer content={item.answer} />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function QACopilot({
  qaList,
  currentQuestion,
  currentAnswer,
  isGenerating,
  isStreamingComplete,
  autoScroll,
}) {
  const copilotEndRef = useRef(null);

  useEffect(() => {
    if (autoScroll) {
      copilotEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [qaList, currentAnswer, autoScroll]);

  return (
    <div className="flex flex-col h-full bg-[#111111] border-l border-[#1A1A1A]">
      
      {/* HEADER – EXACT MATCH WITH INTERVIEWER/CANDIDATE TABS */}
      <div className="h-12 flex items-center justify-between px-6 border-b border-[#242424] bg-[#111111]">

        {/* LEFT: ICON + TITLE */}
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-purple-300" />

          <span className="text-sm font-medium text-white">
            Interview Copilot
          </span>

          <span className="text-xs text-gray-500">
            AI-Powered Assistant
          </span>
        </div>

        {/* RIGHT → ANSWER COUNT */}
        <div className="px-3 py-1 bg-[#1A1A1A] border border-[#2D2D2D] rounded-md text-purple-300 text-xs font-medium">
          {qaList.length} {qaList.length === 1 ? "Answer" : "Answers"}
        </div>
      </div>

      {/* CONTENT AREA */}
      <div className="flex-1 overflow-y-auto p-6">

        {/* EMPTY STATE */}
        {qaList.length === 0 && !currentQuestion ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg flex items-center justify-center mb-4">
              <Sparkles className="w-8 h-8 text-purple-300" />
            </div>

            <h3 className="text-lg font-semibold text-gray-300 mb-1">
              Ready to Assist
            </h3>
            <p className="text-gray-500 max-w-sm">
              AI answers will appear here once the interview begins.
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {/* PREVIOUS QA LIST */}
            {qaList.length > 0 && <QAList qaList={qaList} />}

            {/* CURRENT STREAMING QUESTION */}
            {currentQuestion && (
              <div className="bg-[#181818] border border-[#333] rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-6 h-6 bg-[#222] rounded-md flex items-center justify-center">
                    <span className="text-purple-300 text-sm">❓</span>
                  </div>
                  <span className="text-sm font-bold text-purple-300">
                    CURRENT QUESTION
                  </span>
                </div>

                <p className="text-gray-200 mb-4 leading-relaxed">
                  {currentQuestion}
                </p>

                <div className="bg-[#1f1f1f] border border-[#2a2a2a] rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <MessageCircle className="w-4 h-4 text-purple-300" />
                    <span className="text-sm font-bold text-purple-300">
                      AI ANSWER
                    </span>
                  </div>

                  <div className="text-gray-200 leading-relaxed">
                    {currentAnswer ? (
                      <StreamingAnswer
                        text={currentAnswer}
                        isComplete={isStreamingComplete}
                      />
                    ) : (
                      <div className="flex items-center justify-center py-8">
                        <div className="w-8 h-8 border-4 border-purple-500/40 border-t-purple-500 rounded-full animate-spin" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div ref={copilotEndRef} />
          </div>
        )}
      </div>
    </div>
  );
}
