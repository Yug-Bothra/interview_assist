import React from "react";

export default function ToolsSection() {
  const tools = [
    { 
      title: "AI Mock Interview", 
      desc: "Immersive interview simulation with extensive question banks", 
      img: "https://framerusercontent.com/images/jY8B1tXPxXT2ojQi7WemDeItIEk.png?width=1862&height=1250" 
    },
    { 
      title: "AI Interview Copilot", 
      desc: "Instant, personalized, actionable support during live interviews", 
      img: "https://framerusercontent.com/images/XZi002ckJANbAzVYZZUPAx0lZM.png?width=1062&height=682" 
    },
    { 
      title: "AI Resume Editor", 
      desc: "Generate winning resume that passes screening and stand out", 
      img: "https://framerusercontent.com/images/7Gfj7yHHQyk5gh7BXhQffLCmVtY.png?width=1206&height=722" 
    }
  ];

  return (
    <section className="py-20 bg-white">
      <div className="max-w-6xl mx-auto px-6">
        <p className="text-center text-indigo-600 font-medium mb-4">Interview Made Easy</p>
        <h2 className="text-center text-5xl font-bold mb-4">
          <span className="text-indigo-600">AI-Powered Tools</span> for Every Step of the Job Searching Journey
        </h2>
        <p className="text-center text-gray-600 text-lg mb-16 max-w-3xl mx-auto">
          From building a stellar resume to post-interview insights, we've got you covered before, during, and after the interview
        </p>

        <div className="grid md:grid-cols-3 gap-10 justify-center">
          {tools.map((tool) => (
            <div 
              key={tool.title} 
              className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow flex flex-col w-80 h-[480px]"
            >
              {/* Taller image */}
              <div className="w-full h-64 rounded-xl mb-6 overflow-hidden">
                <img 
                  src={tool.img} 
                  alt={tool.title} 
                  className="w-full h-full object-cover rounded-xl"
                />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">{tool.title}</h3>
              <p className="text-gray-600 mb-4 flex-grow">{tool.desc}</p>
              <button className="px-6 py-2 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition-colors">
                Try now
              </button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
