import React, { useState } from "react";

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState(null);

  const faqs = [
    {
      question: "Can interviewers see or detect the copilot?",
      answer:
        "No, the copilot runs discreetly on your device and is designed to be undetectable. It operates in the background, providing you with real-time suggestions through a separate window or overlay that only you can see. The transcription and AI processing happen locally, ensuring your interview remains natural and professional.",
    },
    {
      question: "Is there a free plan?",
      answer:
        "Yes! We offer a free plan that includes basic features such as limited interview sessions per month, standard AI responses, and support for common interview types. You can upgrade to our Pro plan anytime to unlock unlimited sessions, advanced AI models, industry-specific insights, and priority support.",
    },
    {
      question: "What types of interviews are supported?",
      answer:
        "Our copilot supports all major interview types including behavioral interviews, technical interviews, coding interviews, case studies, system design, product management interviews, and more. Whether it's a phone screen, video interview, or in-person conversation, we've got you covered with tailored guidance for each scenario.",
    },
    {
      question: "What meeting platforms are supported?",
      answer:
        "The copilot works seamlessly with all major video conferencing platforms including Zoom, Microsoft Teams, Google Meet, Webex, Skype, and more. It also supports phone interviews through our mobile app. The platform-agnostic design ensures you can use it regardless of your interviewer's preferred meeting tool.",
    },
    {
      question: "What AI model does interview copilot use?",
      answer:
        "We use state-of-the-art AI models including GPT-4, Claude, and custom-trained models optimized for interview scenarios. Our system intelligently selects the best model based on your interview type, industry, and specific requirements. Pro users get access to the latest and most advanced AI models as soon as they're available.",
    },
    {
      question: "How about coding interviews?",
      answer:
        "Coding interviews are fully supported! The copilot can help you with algorithm questions, data structures, system design, code optimization, debugging, and explaining your thought process. It provides real-time hints, suggests optimal approaches, and helps you articulate your solutions clearly. It supports 20+ programming languages including Python, JavaScript, Java, C++, and more.",
    },
    {
      question: "How do I get started with?",
      answer:
        "Getting started is easy! Simply sign up for a free account, upload your resume and the job description you're applying for, and launch your first interview session. Our onboarding wizard will guide you through the setup process, including microphone configuration and platform selection. You can practice with our AI interviewer or jump straight into a real interview with the copilot assistance.",
    },
  ];

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16">
          {/* Left side - Header */}
          <div>
            <p className="text-indigo-600 font-medium mb-4 text-sm uppercase tracking-wide">
              FAQ
            </p>
            <h2 className="text-5xl font-bold text-gray-900 mb-6">
              Frequently Asked Questions
            </h2>
            <p className="text-lg text-gray-600 leading-relaxed">
              We address common queries, demystify intricacies, and provide
              insights to guide you through our services.
            </p>
          </div>

          {/* Right side - FAQ Items */}
          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <div
                key={index}
                className="border border-gray-200 rounded-lg overflow-hidden transition-all duration-300 hover:border-indigo-300"
              >
                <button
                  onClick={() => toggleFAQ(index)}
                  className="w-full flex items-center justify-between p-6 text-left bg-white hover:bg-gray-50 transition-colors"
                >
                  <span className="text-lg font-semibold text-gray-900 pr-4">
                    {faq.question}
                  </span>
                  <span
                    className={`flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-full bg-indigo-50 text-indigo-600 transition-transform duration-300 ${
                      openIndex === index ? "rotate-45" : ""
                    }`}
                  >
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 4v16m8-8H4"
                      />
                    </svg>
                  </span>
                </button>

                <div
                  className={`overflow-hidden transition-all duration-300 ${
                    openIndex === index ? "max-h-96" : "max-h-0"
                  }`}
                >
                  <div className="p-6 pt-0 text-gray-600 leading-relaxed">
                    {faq.answer}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}