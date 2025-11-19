import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../Landing/Navbar";
import Footer from "../../Landing/Footer";

export default function Blog() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");

  const blogPosts = [
    {
      id: 1,
      date: "Apr 8, 2025",
      title: "30 Most Common Attitude Questions Interview Questions You Should Prepare For",
      image: "https://images.unsplash.com/photo-1521737711867-e3b97375f902?w=800&h=500&fit=crop",
      link: "#"
    },
    {
      id: 2,
      date: "Jan 17, 2025",
      title: "First Round of Interviews More Difficult Than Expected?",
      image: "https://images.unsplash.com/photo-1588196749597-9ff075ee6b5b?w=800&h=500&fit=crop",
      link: "#"
    },
    {
      id: 3,
      date: "Feb 25, 2025",
      title: "Is Final Round AI Legit? Unveiling the Truth About AI-Powered Interview Prep",
      image: "https://images.unsplash.com/photo-1551434678-e076c223a692?w=800&h=500&fit=crop",
      link: "#"
    },
    {
      id: 4,
      date: "Jun 28, 2024",
      title: "Why Should You Join Us? Mastering the Interview Question with Interview Copilot",
      image: "https://images.unsplash.com/photo-1552664730-d307ca884978?w=800&h=500&fit=crop",
      link: "#"
    },
    {
      id: 5,
      date: "Oct 21, 2024",
      title: "Top Features to Look for in an AI Interview Copilot",
      image: "https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?w=800&h=500&fit=crop",
      link: "#"
    },
    {
      id: 6,
      date: "Feb 10, 2025",
      title: "High School Student Resume Secrets: Unlocking Your Path to a Successful Internship Application",
      image: "https://images.unsplash.com/photo-1556761175-b413da4baf72?w=800&h=500&fit=crop",
      link: "#"
    }
  ];

  const filteredPosts = blogPosts.filter(post =>
    post.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">

      {/* Navbar */}
      <Navbar />

      {/* Hero Section */}
      <div className="pt-32 pb-16 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
            Interview Tips & More
          </h1>
          <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed">
            Offering tips, expert strategies, and tools to help you master every aspect of the job interview process and achieve career success
          </p>

          {/* Search Bar */}
          <div className="max-w-2xl mx-auto relative">
            <input
              type="text"
              placeholder="Search resources..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-6 py-4 pr-12 rounded-full border-2 border-gray-200 focus:border-indigo-500 focus:outline-none text-gray-700 text-lg transition-all"
            />
            <svg
              className="w-6 h-6 text-gray-400 absolute right-5 top-1/2 transform -translate-y-1/2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Blog Grid */}
      <div className="max-w-7xl mx-auto px-6 pb-20">
        <div className="grid md:grid-cols-2 gap-8">
          {filteredPosts.map((post) => (
            <div
              key={post.id}
              className="bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300 cursor-pointer border border-gray-100 group"
            >
              <div className="relative overflow-hidden">
                <img
                  src={post.image}
                  alt={post.title}
                  className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="p-8">
                <p className="text-sm text-gray-500 mb-3">{post.date}</p>
                <h3 className="text-2xl font-bold text-gray-900 mb-4 group-hover:text-indigo-600 transition-colors">
                  {post.title}
                </h3>
                <a
                  href={post.link}
                  className="inline-flex items-center text-indigo-600 font-semibold hover:text-indigo-700 transition-colors"
                >
                  Read now
                  <svg
                    className="w-5 h-5 ml-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M17 8l4 4m0 0l-4 4m4-4H3"
                    />
                  </svg>
                </a>
              </div>
            </div>
          ))}
        </div>

        {filteredPosts.length === 0 && (
          <div className="text-center py-20">
            <p className="text-xl text-gray-500">No blog posts found matching your search.</p>
          </div>
        )}
      </div>

      {/* CTA Section */}
      <div className="bg-gray-900 py-16">
        <div className="max-w-4xl mx-auto text-center px-6">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
            Interview Copilot: Your AI-Powered Personalized Cheatsheet
          </h2>
          <button
            onClick={() => navigate("/sign-in")}
            className="px-8 py-4 bg-yellow-400 text-gray-900 rounded-full font-bold text-lg hover:bg-yellow-300 transition-colors shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300"
          >
            Start Free Trial
          </button>
        </div>
      </div>

      {/* Footer */}
      <Footer />

    </div>
  );
}
