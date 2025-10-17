import React from 'react';
import { Lock, Code, FileText, Phone, Video, Database, Megaphone } from 'lucide-react';

const domains = [
  {
    name: 'General Interview',
    description: 'Behavioral, Conceptual, and Technical Questions All-In-One',
    icon: FileText,
    available: true,
    color: 'from-yellow-500 to-yellow-600'
  },
  {
    name: 'Coding Interview',
    description: 'One-Click Coding Interpreter Plugin for 30+ Programming Languages',
    icon: Code,
    available: false,
    color: 'from-green-500 to-green-600'
  },
  {
    name: 'Online Assessment',
    description: 'Crack Online Tests Instantly with Powerful LLM Vision Technology',
    icon: FileText,
    available: false,
    color: 'from-orange-500 to-orange-600'
  },
  {
    name: 'Phone Interview',
    description: 'Real-Time Interview Copilot for Calls on Another Device',
    icon: Phone,
    available: false,
    color: 'from-blue-500 to-blue-600'
  },
  {
    name: 'Professional Meeting',
    description: 'AI Copilot and Meeting Notes For Your Professional Meetings',
    icon: Video,
    available: false,
    color: 'from-purple-500 to-purple-600'
  },
  {
    name: 'Software Engineering',
    description: 'Algorithms, System Design Patterns, DevOps Standards and more',
    icon: Code,
    available: false,
    color: 'from-indigo-500 to-indigo-600'
  },
  {
    name: 'Data Science & Analytics',
    description: 'SQL, Statistical Methods, Machine Learning Models, Data Visualization and more',
    icon: Database,
    available: false,
    color: 'from-violet-500 to-violet-600'
  },
  {
    name: 'Marketing',
    description: 'Marketing Strategies, Digital Marketing, Brand Management and more',
    icon: Megaphone,
    available: false,
    color: 'from-pink-500 to-pink-600'
  }
];

export default function InterviewDomainSelection({ onSelect }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        {domains.map((domain, index) => {
          const IconComponent = domain.icon;
          return (
            <button
              key={index}
              onClick={() => domain.available && onSelect(domain.name)}
              disabled={!domain.available}
              className={`bg-[#14141b] border border-gray-800 rounded-xl p-6 text-left transition-all relative ${
                domain.available
                  ? 'hover:border-purple-500/50 hover:bg-gray-900/50 cursor-pointer'
                  : 'opacity-60 cursor-not-allowed'
              }`}
            >
              {!domain.available && (
                <div className="absolute top-4 right-4">
                  <Lock className="w-4 h-4 text-gray-500" />
                </div>
              )}
              <div className={`w-12 h-12 bg-gradient-to-br ${domain.color} rounded-lg flex items-center justify-center mb-4 opacity-80`}>
                <IconComponent className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-white font-semibold mb-2">{domain.name}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{domain.description}</p>
            </button>
          );
        })}
      </div>

      <div className="flex justify-center gap-2">
        <button className="w-10 h-10 rounded-full bg-purple-500 text-white flex items-center justify-center font-semibold">
          1
        </button>
        <button className="w-10 h-10 rounded-full bg-gray-800 text-gray-400 flex items-center justify-center font-semibold hover:bg-gray-700">
          2
        </button>
      </div>
    </div>
  );
}