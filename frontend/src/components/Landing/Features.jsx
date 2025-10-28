import React from "react";

export default function Features() {
  const features = [
    {
      icon: "✨",
      label: "AI-Powered Responses",
      title: "Personalized interview support tailored to resume, role, & company",
      description:
        "Get customized guidance designed to align with your skills, job description, and the company's expectations",
      image:
        "https://framerusercontent.com/images/GdYMwd4ZE6zFiZly0ipERUDxsnA.png?scale-down-to=1024&width=1298&height=804",
      color: "indigo",
    },
    {
      icon: "✨",
      label: "Support All Interview Scenarios",
      title: "From behavioral questions to coding interviews",
      description:
        "Tackle all scenarios confidently with real-time guidance and tailored responses",
      image:
        "https://framerusercontent.com/images/UqUDc8ONAON8HuhS5aLpMtjxug.gif?width=500&height=400",
      color: "purple",
    },
    {
      icon: "✨",
      label: "Support 25+ Languages",
      title: "Multilingual transcription and real-time interview guidance",
      description:
        "Get real-time, personalized interview support in any language",
      image:
        "https://framerusercontent.com/images/pCbPcBMqDxcXgPZ5NEHMCCAy7Y.png?scale-down-to=1024&width=1266&height=1256",
      color: "blue",
    },
  ];

  return (
    <section className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        {features.map((feature, index) => (
          <div
            key={index}
            className={`flex flex-col ${
              index % 2 === 0 ? "md:flex-row" : "md:flex-row-reverse"
            } items-center gap-12 mb-24`}
          >
            <div className="flex-1">
              <div className="inline-block px-4 py-2 bg-indigo-50 text-indigo-600 rounded-full text-sm font-medium mb-4">
                {feature.icon} {feature.label}
              </div>
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                {feature.title}
              </h2>
              <p className="text-lg text-gray-600 mb-6">
                {feature.description}
              </p>
              <button className="px-6 py-3 bg-lime-300 text-gray-900 rounded-lg font-semibold hover:bg-lime-400 transition-colors">
                Start Free Trial
              </button>
            </div>

            <div className="flex-1">
              <div
                className={`w-full h-96 bg-gradient-to-br ${
                  feature.color === "indigo"
                    ? "from-indigo-100 to-purple-100"
                    : feature.color === "purple"
                    ? "from-purple-100 to-indigo-100"
                    : "from-blue-100 to-indigo-100"
                } rounded-2xl shadow-xl flex items-center justify-center overflow-hidden`}
              >
                <img
                  src={feature.image}
                  alt={feature.label}
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Industry Expertise Section */}
      <div className="max-w-7xl mx-auto px-6 mt-32">
        <div className="text-center mb-12">
          <p className="text-indigo-600 font-medium mb-4">
            Upgrade to Pro to Unlock All Add-Ons
          </p>
          <h2 className="text-5xl font-bold mb-4">
            <span className="text-indigo-600">AI Interview Copilot</span> with
            Expertise Across 100+ Industries
          </h2>
          <p className="text-gray-600 text-lg max-w-4xl mx-auto">
            Supercharge your AI Interview Copilot with insights from 100+
            industries, including trends, news, tailored interview formats, and
            job opportunities
          </p>
        </div>

        {/* Category Tabs */}
        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {[
            "Software",
            "Product",
            "Consulting",
            "Data",
            "Marketing",
            "Finance",
          ].map((cat) => (
            <button
              key={cat}
              className={`px-6 py-2.5 rounded-full font-medium transition-colors ${
                cat === "Product"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Job Roles Grid */}
        <div className="grid md:grid-cols-3 gap-6">
          {[
            "Product Manager",
            "Product Owner",
            "Product Analyst",
            "Technical Product Manager",
            "Project Manager",
            "UX/Product Researcher",
            "Product Designer",
            "Product Marketing Manager",
            "Growth Product Manager",
          ].map((role) => (
            <div
              key={role}
              className="bg-indigo-50 rounded-xl p-6 hover:shadow-lg transition-shadow"
            >
              <h3 className="text-xl font-bold text-gray-900 mb-4">{role}</h3>
              <button className="px-4 py-2 bg-white rounded-lg font-medium text-gray-700 hover:bg-gray-50 transition-colors shadow-sm">
                Try now
              </button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
