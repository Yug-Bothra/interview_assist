import React, { useState } from "react";

export default function Pricing() {
  const [billingPeriod, setBillingPeriod] = useState("quarterly");
  
  const modernPlans = [
    {
      badge: "FREE",
      badgeColor: "bg-indigo-600 text-white",
      title: "For Job Hunters",
      price: "0",
      originalPrice: null,
      period: "/month",
      features: [
        "15-min Copilot Session x 3",
        "AI Mock Interviews x 5",
        "AI Resume Builder",
        "Live Chat Support"
      ],
      buttonText: "Free",
      buttonStyle: "border-2 border-gray-900 text-gray-900 hover:bg-gray-50",
      cardStyle: "bg-white border-2 border-gray-200"
    },
    {
      badge: "STANDARD",
      badgeColor: "bg-indigo-600 text-white",
      title: "For Active Interviewees",
      price: billingPeriod === "yearly" ? "14.45" : billingPeriod === "quarterly" ? "34" : "38.25",
      originalPrice: billingPeriod === "yearly" ? "17" : billingPeriod === "quarterly" ? "40" : "45",
      period: "/month",
      features: [
        "60-min Copilot Session x 5",
        "Unlimited Mock Interviews",
        "Question Banks",
        "Interview Report",
        "15 + AI Career Tools",
        "Everything in Free"
      ],
      buttonText: "Get Started",
      buttonStyle: "border-2 border-gray-900 text-gray-900 hover:bg-gray-50",
      cardStyle: "bg-white border-2 border-gray-200"
    },
    {
      badge: "PRO",
      badgeColor: "bg-yellow-400 text-gray-900",
      bestValue: true,
      title: "For Offer Seekers",
      price: billingPeriod === "yearly" ? "25.5" : billingPeriod === "quarterly" ? "51" : "59.5",
      originalPrice: billingPeriod === "yearly" ? "30" : billingPeriod === "quarterly" ? "60" : "70",
      period: "/month",
      features: [
        "Unlimited 90-min Copilot Sessions",
        "Unlimited Mock Interviews",
        "Coding Copilot",
        "Online Assessment Copilot",
        "Industry Knowledge Add-Ons",
        "Question Banks",
        "Interview Report",
        "Everything in Standard"
      ],
      buttonText: "Get Started",
      buttonStyle: "bg-white text-indigo-600 hover:bg-gray-50",
      cardStyle: "bg-indigo-600 text-white"
    }
  ];

  return (
    <section id="pricing" className="min-h-screen py-16 bg-gradient-to-br from-gray-50 via-purple-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-3 text-gray-900">
            All-Inclusive Subscription:
          </h2>
          <h3 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-6 text-gray-900">
            No Limits, No Extra Fees
          </h3>
          <p className="text-gray-600 mb-8 text-sm sm:text-base">
            Get started with a free plan (No Credit Card Needed)
          </p>
          
          {/* Billing Period Toggle */}
          <div className="inline-flex items-center gap-2 sm:gap-3 bg-white rounded-full p-1.5 sm:p-2 shadow-lg">
            <button
              onClick={() => setBillingPeriod("yearly")}
              className={`relative px-4 sm:px-6 py-2 rounded-full font-semibold text-sm sm:text-base transition-all ${
                billingPeriod === "yearly" 
                  ? "bg-indigo-600 text-white" 
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              Yearly
              {billingPeriod === "yearly" && (
                <span className="absolute -top-2 -right-2 bg-yellow-400 text-gray-900 text-xs font-bold px-2 py-0.5 rounded-full whitespace-nowrap">
                  63% OFF
                </span>
              )}
            </button>
            <button
              onClick={() => setBillingPeriod("quarterly")}
              className={`px-4 sm:px-6 py-2 rounded-full font-semibold text-sm sm:text-base transition-all ${
                billingPeriod === "quarterly" 
                  ? "bg-indigo-600 text-white" 
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              Quarterly
            </button>
            <button
              onClick={() => setBillingPeriod("monthly")}
              className={`px-4 sm:px-6 py-2 rounded-full font-semibold text-sm sm:text-base transition-all ${
                billingPeriod === "monthly" 
                  ? "bg-indigo-600 text-white" 
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              Monthly
            </button>
          </div>
        </div>

        {/* Pricing Cards */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8 max-w-6xl mx-auto mb-8">
          {modernPlans.map((plan, idx) => (
            <div
              key={idx}
              className={`relative rounded-3xl p-6 sm:p-8 transition-all ${
                plan.cardStyle
              } ${plan.bestValue ? "shadow-2xl ring-2 ring-indigo-400" : "shadow-lg hover:shadow-xl"}`}
            >
              {/* Badge and Best Value */}
              <div className="flex items-center justify-between mb-4">
                <span className={`${plan.badgeColor} px-3 sm:px-4 py-1 rounded-full text-xs font-bold`}>
                  {plan.badge}
                </span>
                {plan.bestValue && (
                  <span className="bg-white text-indigo-600 px-3 py-1 rounded-full text-xs font-bold">
                    Best Value
                  </span>
                )}
              </div>

              {/* Title */}
              <h3 className={`text-base sm:text-lg mb-4 ${plan.cardStyle.includes("bg-indigo-600") ? "text-white" : "text-gray-600"}`}>
                {plan.title}
              </h3>

              {/* Price */}
              <div className="mb-6">
                {plan.originalPrice && (
                  <span className={`text-xl sm:text-2xl line-through mr-2 ${
                    plan.cardStyle.includes("bg-indigo-600") ? "text-white/70" : "text-gray-400"
                  }`}>
                    ${plan.originalPrice}
                  </span>
                )}
                <span className={`text-4xl sm:text-5xl font-bold ${
                  plan.cardStyle.includes("bg-indigo-600") ? "text-white" : "text-gray-900"
                }`}>
                  ${plan.price}
                </span>
                <span className={`text-base sm:text-lg ml-1 ${
                  plan.cardStyle.includes("bg-indigo-600") ? "text-white/80" : "text-gray-600"
                }`}>
                  {plan.period}
                </span>
              </div>

              {/* Features */}
              <ul className="space-y-3 mb-8">
                {plan.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-2 sm:gap-3">
                    <svg 
                      className={`w-5 h-5 mt-0.5 flex-shrink-0 ${
                        plan.cardStyle.includes("bg-indigo-600") ? "text-white" : "text-indigo-600"
                      }`} 
                      fill="currentColor" 
                      viewBox="0 0 20 20"
                    >
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span className={`text-sm ${
                      plan.cardStyle.includes("bg-indigo-600") ? "text-white" : "text-gray-700"
                    }`}>
                      {feature}
                    </span>
                  </li>
                ))}
              </ul>

              {/* CTA Button */}
              <button className={`w-full py-3 rounded-xl font-semibold transition-all text-sm sm:text-base ${plan.buttonStyle}`}>
                {plan.buttonText} →
              </button>
            </div>
          ))}
        </div>

        {/* Footer Note */}
        <div className="text-center mb-16">
          <p className="text-sm text-gray-500">
            Offer End Soon: 15% Off Your First Purchase
          </p>
        </div>

        {/* FAQ Section */}
        <div className="max-w-4xl mx-auto mb-16">
          <div className="text-center mb-8">
            <p className="text-indigo-600 text-sm font-semibold mb-2">Got Questions?</p>
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">
              Learn more about our pricing
            </h2>
          </div>

          <div className="space-y-4">
            <FAQItem
              question="What's included in the free plan?"
              answer="The free plan gives you 3 Copilot sessions (15 minutes each)—perfect for quick phone screens or trying things out. You'll also get access to core features like mock interviews, question banks, and over 20 free AI tools, including our AI resume editor."
              defaultOpen={true}
            />
            <FAQItem
              question="Can I cancel or change my plan anytime?"
              answer="Yes! You can cancel or change your plan at any time. No questions asked, no hassle. Your subscription will remain active until the end of your current billing period."
            />
            <FAQItem
              question="Do you offer team or student discounts?"
              answer="Yes, we offer special discounts for teams and students. Please contact our support team for more information about bulk pricing and educational discounts."
            />
            <FAQItem
              question="What is the refund policy?"
              answer="We offer a 7-day money-back guarantee on all paid plans. If you're not satisfied within the first 7 days, we'll provide a full refund, no questions asked."
            />
            <FAQItem
              question="What payment methods do you accept?"
              answer="We accept all major credit cards (Visa, Mastercard, American Express), debit cards, and PayPal. All payments are processed securely through our payment provider."
            />
          </div>
        </div>

        {/* CTA Banner */}
        <div className="bg-gray-900 rounded-3xl p-8 sm:p-12 text-center text-white">
          <h2 className="text-2xl sm:text-3xl font-bold mb-4">
            Interview Copilot: Your AI-Powered Personalized Cheatsheet
          </h2>
          <button className="bg-yellow-400 text-gray-900 px-8 py-3 rounded-xl font-bold hover:bg-yellow-300 transition-all">
            Start Free Trial
          </button>
        </div>
      </div>
    </section>
  );
}

function FAQItem({ question, answer, defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-white rounded-2xl border-2 border-gray-200">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 sm:px-8 py-5 sm:py-6 flex items-center justify-between text-left hover:bg-gray-50 transition-colors"
      >
        <span className="text-base sm:text-lg font-semibold text-gray-900 pr-4">
          {question}
        </span>
        <svg
          className={`w-5 h-5 flex-shrink-0 text-gray-600 transition-transform duration-200 ${
            isOpen ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div 
        className={`overflow-hidden transition-all duration-300 ease-in-out ${
          isOpen ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <div className="px-6 sm:px-8 pb-5 sm:pb-6">
          <p className="text-gray-600 text-sm sm:text-base leading-relaxed">
            {answer}
          </p>
        </div>
      </div>
    </div>
  );
}