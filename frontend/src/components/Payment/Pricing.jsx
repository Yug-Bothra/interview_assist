// --------------------------------------------------
// src/components/Payment/Pricing.jsx
// FINAL — PayU Same-Tab Payment Redirect (Popup-Proof)
// --------------------------------------------------

import React, { useState } from "react";
import Navbar from "../Landing/Navbar";
import Footer from "../Landing/Footer";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../Auth/AuthContext";

export default function Pricing() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [billingPeriod, setBillingPeriod] = useState("quarterly");
  const [loading, setLoading] = useState(false);

  // ---------------------------------------------------------
  // START PAYMENT — SAME TAB REDIRECT
  // ---------------------------------------------------------
  const startPayment = async (plan) => {
    try {
      if (!user) {
        alert("Please login to purchase a plan.");
        return navigate("/sign-in");
      }

      setLoading(plan.key);

      const payload = {
        user_id: user.id,
        plan: plan.key,
        billing_period: billingPeriod,
        amount: plan.amountINR.toString(),
        firstname: user.user_metadata?.full_name || "User",
        email: user.email,
        phone: "9999999999",
      };

      // BACKEND URL
      const API_BASE = import.meta.env.DEV
        ? "http://localhost:8000"
        : import.meta.env.VITE_API_BASE_URL;

      const res = await fetch(`${API_BASE}/payment/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!data?.form) {
        console.error("PAYU ERROR:", data);
        alert("Payment server error.");
        setLoading(false);
        return;
      }

      // 🔥 SAME TAB REDIRECT — ZERO POPUP ISSUES
      const newDoc = document.open("text/html", "replace");
      newDoc.write(data.form);
      newDoc.close();

    } catch (err) {
      console.error("PAYMENT ERROR:", err);
      alert("Unable to process payment.");
      setLoading(false);
    }
  };

  // ---------------------------------------------------------
  // ALL PRICING PLANS
  // ---------------------------------------------------------
  const modernPlans = [
    {
      key: "free",
      badge: "FREE",
      badgeColor: "bg-indigo-600 text-white",
      title: "For Job Hunters",
      amountINR: 0,
      price: "0",
      originalPrice: null,
      period: "/month",
      features: [
        "15-min Copilot Session x 3",
        "AI Mock Interviews x 5",
        "AI Resume Builder",
        "Live Chat Support",
      ],
      buttonText: "Free",
      buttonStyle: "border-2 border-gray-900 text-gray-900 hover:bg-gray-50",
      cardStyle: "bg-white border-2 border-gray-200",
    },
    {
      key: "standard",
      badge: "STANDARD",
      badgeColor: "bg-indigo-600 text-white",
      title: "For Active Interviewees",
      amountINR:
        billingPeriod === "yearly"
          ? 1445
          : billingPeriod === "quarterly"
          ? 3400
          : 3825,
      price:
        billingPeriod === "yearly"
          ? "14.45"
          : billingPeriod === "quarterly"
          ? "34"
          : "38.25",
      originalPrice:
        billingPeriod === "yearly"
          ? "17"
          : billingPeriod === "quarterly"
          ? "40"
          : "45",
      period: "/month",
      features: [
        "60-min Copilot Session x 5",
        "Unlimited Mock Interviews",
        "Question Banks",
        "Interview Report",
        "15+ AI Career Tools",
        "Everything in Free",
      ],
      buttonText: "Get Started",
      buttonStyle: "border-2 border-gray-900 text-gray-900 hover:bg-gray-50",
      cardStyle: "bg-white border-2 border-gray-200",
    },
    {
      key: "pro",
      badge: "PRO",
      badgeColor: "bg-yellow-400 text-gray-900",
      title: "For Offer Seekers",
      amountINR:
        billingPeriod === "yearly"
          ? 2550
          : billingPeriod === "quarterly"
          ? 5100
          : 5950,
      price:
        billingPeriod === "yearly"
          ? "25.5"
          : billingPeriod === "quarterly"
          ? "51"
          : "59.5",
      originalPrice:
        billingPeriod === "yearly"
          ? "30"
          : billingPeriod === "quarterly"
          ? "60"
          : "70",
      period: "/month",
      features: [
        "Unlimited 90-min Copilot Sessions",
        "Unlimited Mock Interviews",
        "Coding Copilot",
        "Online Assessment Copilot",
        "Industry Knowledge Add-Ons",
        "Question Banks",
        "Interview Report",
        "Everything in Standard",
      ],
      buttonText: "Go Pro",
      buttonStyle: "bg-white text-indigo-600 hover:bg-gray-50",
      cardStyle: "bg-indigo-600 text-white",
      bestValue: true,
    },
  ];

  // ---------------------------------------------------------
  // RENDER UI
  // ---------------------------------------------------------
  return (
    <>
      <Navbar />

      <section className="min-h-screen py-16 bg-gradient-to-br from-gray-50 via-purple-50 to-indigo-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">

          {/* TITLE */}
          <div className="text-center mb-10">
            <h2 className="text-4xl font-bold text-gray-900">
              All-Inclusive Subscription
            </h2>
            <p className="text-gray-600 mt-2">Choose the plan that fits you best.</p>
          </div>

          {/* BILLING PERIOD TOGGLE */}
          <div className="flex justify-center mb-12">
            <div className="flex bg-white rounded-full shadow p-2 space-x-3">
              {["yearly", "quarterly", "monthly"].map((period) => (
                <button
                  key={period}
                  onClick={() => setBillingPeriod(period)}
                  className={`px-6 py-2 rounded-full font-semibold ${
                    billingPeriod === period
                      ? "bg-indigo-600 text-white"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  {period.charAt(0).toUpperCase() + period.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* PRICING CARDS */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {modernPlans.map((plan, idx) => (
              <div
                key={idx}
                className={`rounded-3xl p-8 transition-all ${plan.cardStyle} ${
                  plan.bestValue ? "shadow-2xl ring-2 ring-indigo-400" : "shadow-lg"
                }`}
              >
                {/* BADGE */}
                <span className={`${plan.badgeColor} px-4 py-1 rounded-full text-xs font-bold`}>
                  {plan.badge}
                </span>

                <h3 className="text-xl mt-4 mb-2">{plan.title}</h3>

                {/* PRICE */}
                <div className="mb-6">
                  {plan.originalPrice && (
                    <span className="line-through text-gray-400 text-xl mr-2">
                      ${plan.originalPrice}
                    </span>
                  )}
                  <span className="text-5xl font-bold">{plan.price}</span>
                  <span className="ml-1 text-gray-600">{plan.period}</span>
                </div>

                {/* FEATURES */}
                <ul className="space-y-3 mb-8">
                  {plan.features.map((f, idf) => (
                    <li key={idf} className="flex items-start gap-3">
                      <span className="text-indigo-600">✔</span>
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>

                {/* PAYMENT BUTTON */}
                <button
                  disabled={loading === plan.key}
                  onClick={() => startPayment(plan)}
                  className={`w-full py-3 rounded-xl font-semibold ${plan.buttonStyle} ${
                    loading === plan.key ? "opacity-50 cursor-not-allowed" : ""
                  }`}
                >
                  {loading === plan.key ? "Processing…" : plan.buttonText} →
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
