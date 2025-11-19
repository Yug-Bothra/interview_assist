import React from "react";
import { useNavigate } from "react-router-dom";

// Import Navbar, Footer, ToolsSection
import Navbar from "../../Landing/Navbar";
import Footer from "../../Landing/Footer";
import ToolsSection from "../../Landing/ToolsSection";

export default function ResumeBuilder() {
  const navigate = useNavigate();

  const resumeFeatures = [
    {
      icon: "✍️",
      title: "AI Writing",
      description: "Share your thoughts, and let our AI optimize the language for you",
    },
    {
      icon: "📄",
      title: "Resume Template",
      description: "Professional template that highlights your strengths",
    },
    {
      icon: "🎯",
      title: "ATS Optimized",
      description: "Highlight key skills and experience to pass resume screening",
    },
  ];

  const tools = [
    {
      icon: "✏️",
      title: "AI Cover Letter",
      description: "Generate a cover letter that grabs recruiters' attention.",
      iconBg: "bg-indigo-50",
      iconColor: "text-indigo-600",
    },
    {
      icon: "📋",
      title: "Resume Optimizer",
      description: "Optimize your resume to receive more job interviews.",
      iconBg: "bg-indigo-50",
      iconColor: "text-indigo-600",
    },
    {
      icon: "🔍",
      title: "ATS Checker",
      description: "Check if your resume can pass resume screening system.",
      iconBg: "bg-indigo-50",
      iconColor: "text-indigo-600",
    },
    {
      icon: "⭐",
      title: "Resume Rater",
      description: "Get a score and feedback to improve your resume.",
      iconBg: "bg-indigo-50",
      iconColor: "text-indigo-600",
    },
    {
      icon: "📝",
      title: "Resume Formatter",
      description: "Reformat your resume for clean, professional layouts.",
      iconBg: "bg-indigo-50",
      iconColor: "text-indigo-600",
    },
    {
      icon: "🔧",
      title: "Resume Fixer",
      description: "Quickly fix resume issues like formatting, grammar, etc.",
      iconBg: "bg-indigo-50",
      iconColor: "text-indigo-600",
    },
    {
      icon: "💼",
      title: "Job Description Generator",
      description: "Craft perfect job descriptions for your resume.",
      iconBg: "bg-indigo-50",
      iconColor: "text-indigo-600",
    },
    {
      icon: "✉️",
      title: "Thank-you Email Generator",
      description: "Write a polished thank-you email in seconds.",
      iconBg: "bg-indigo-50",
      iconColor: "text-indigo-600",
    },
  ];

  const reviews = [
    {
      name: "Sarah Johnson",
      role: "Software Engineer",
      avatar: "SJ",
      rating: 5,
      text: "The AI Resume Builder helped me land my dream job at a top tech company! The ATS optimization feature was a game-changer.",
      color: "bg-indigo-500",
    },
    {
      name: "Michael Chen",
      role: "Product Manager",
      avatar: "MC",
      rating: 5,
      text: "I went from 2 interviews per month to 8! The resume templates are professional and the AI writing suggestions are spot-on.",
      color: "bg-purple-500",
    },
    {
      name: "Emily Rodriguez",
      role: "Data Analyst",
      avatar: "ER",
      rating: 5,
      text: "Best investment in my career. The cover letter generator alone saved me hours of work. Highly recommend!",
      color: "bg-pink-500",
    },
    {
      name: "David Kim",
      role: "Marketing Manager",
      avatar: "DK",
      rating: 5,
      text: "The ATS checker feature is incredible. My resume now passes through applicant tracking systems with ease. Got 3 offers in a month!",
      color: "bg-blue-500",
    },
    {
      name: "Jessica Williams",
      role: "UX Designer",
      avatar: "JW",
      rating: 5,
      text: "Clean, modern templates and intelligent AI suggestions. This tool completely transformed my job search experience.",
      color: "bg-green-500",
    },
    {
      name: "Alex Thompson",
      role: "Business Analyst",
      avatar: "AT",
      rating: 5,
      text: "From zero callbacks to multiple interview invites. The resume optimizer is worth every penny. Thank you!",
      color: "bg-orange-500",
    },
  ];

  const renderStars = (rating) => {
    return (
      <div className="flex gap-1">
        {[...Array(rating)].map((_, i) => (
          <svg
            key={i}
            className="w-5 h-5 text-yellow-400 fill-current"
            viewBox="0 0 20 20"
          >
            <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
          </svg>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Navbar */}
      <Navbar />

      {/* Hero Section */}
      <section className="py-28 bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 mt-16">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h1 className="text-6xl font-bold text-gray-900 mb-6">
            AI Resume Builder
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Harnesses the power of AI to create, optimize, and personalize your
            resume in no-time
          </p>
          <button className="px-8 py-4 bg-gray-800 text-white rounded-lg font-semibold text-lg hover:bg-gray-900 transition-colors shadow-lg">
            Free Forever
          </button>
        </div>
      </section>

      {/* Get 20x More Interview Invites Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <p className="text-indigo-600 font-medium mb-4">
                Interview Made Easy
              </p>
              <h2 className="text-5xl font-bold text-gray-900 mb-6">
                Get <span className="text-indigo-600">20x</span> More Interview
                Invites
              </h2>
              <p className="text-xl text-gray-600">
                Tired of getting ghosted? Get 10x more interviews with AI Resume
                Builder
              </p>
            </div>

            <div className="space-y-6">
              {resumeFeatures.map((feature, index) => (
                <div
                  key={index}
                  className="bg-white border border-gray-200 rounded-2xl p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-indigo-600 rounded-full flex items-center justify-center text-2xl flex-shrink-0">
                      {feature.icon}
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-gray-900 mb-2">
                        {feature.title}
                      </h3>
                      <p className="text-gray-600">{feature.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Tools Section - Free Toolkit for Landing Jobs */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <p className="text-indigo-600 font-medium mb-4">Hoot Tools</p>
            <h2 className="text-5xl font-bold text-gray-900 mb-6">
              Free Toolkit for Landing Jobs This Season
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Tools that do the heavy lifting so you can focus on showing up
              strong
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {tools.map((tool, index) => (
              <div
                key={index}
                className="bg-white rounded-2xl p-8 hover:shadow-xl transition-shadow border border-gray-100"
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`w-14 h-14 ${tool.iconBg} rounded-xl flex items-center justify-center text-3xl flex-shrink-0`}
                  >
                    {tool.icon}
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-3">
                      {tool.title}
                    </h3>
                    <p className="text-gray-600 text-lg">{tool.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ToolsSection Component */}
      <ToolsSection />

      {/* Reviews Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold text-gray-900 mb-4">
              Loved by Job Seekers Worldwide
            </h2>
            <p className="text-xl text-gray-600">
              Join thousands who've transformed their job search
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {reviews.map((review, index) => (
              <div
                key={index}
                className="bg-white border border-gray-200 rounded-2xl p-6 hover:shadow-xl transition-shadow"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div
                    className={`w-12 h-12 ${review.color} rounded-full flex items-center justify-center text-white font-bold text-lg`}
                  >
                    {review.avatar}
                  </div>
                  <div>
                    <h4 className="font-bold text-gray-900">{review.name}</h4>
                    <p className="text-sm text-gray-600">{review.role}</p>
                  </div>
                </div>
                <div className="mb-4">{renderStars(review.rating)}</div>
                <p className="text-gray-700 leading-relaxed">{review.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
}