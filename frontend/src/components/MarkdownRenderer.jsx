import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Check } from "lucide-react";

export default function MarkdownRenderer({ content }) {
  const [copiedIndex, setCopiedIndex] = useState(null);

  if (!content) return null;

  const handleCopy = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 500);
  };

  let codeBlockIndex = 0;

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeRaw]}
      components={{
        code({ inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");

          // ----------- BLOCK CODE (GPT STYLE) ------------
          if (!inline && match) {
            const language = match[1];
            const currentIndex = codeBlockIndex++;
            const codeText = String(children);

            return (
              <div className="bg-[#1e1e1e] rounded-xl border border-[#333] my-4 overflow-hidden">
                
                {/* Header */}
                <div className="bg-[#151515] px-3 py-2 border-b border-[#333] flex items-center justify-between">
                  <span className="text-xs text-[#9cdcfe] font-semibold uppercase">
                    {language}
                  </span>

                  <button
                    onClick={() => handleCopy(codeText, currentIndex)}
                    className="text-xs text-gray-300 hover:text-white px-2 py-1 rounded flex items-center gap-1"
                  >
                    {copiedIndex === currentIndex ? (
                      <>
                        <Check size={14} />
                        Copied
                      </>
                    ) : (
                      "Copy"
                    )}
                  </button>
                </div>

                {/* Code */}
                <SyntaxHighlighter
                  language={language}
                  style={oneDark}
                  customStyle={{
                    margin: 0,
                    padding: "14px",
                    background: "#1e1e1e",
                    fontSize: "14px",
                  }}
                >
                  {codeText.replace(/\n$/, "")}
                </SyntaxHighlighter>
              </div>
            );
          }

          // ----------- INLINE CODE -----------
          return (
            <code className="bg-[#2a2a2a] px-2 py-1 rounded text-sm">
              {children}
            </code>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}