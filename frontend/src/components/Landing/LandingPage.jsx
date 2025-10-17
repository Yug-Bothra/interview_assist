import React from "react";
import Hero from "./Hero";
import Features from "./Features";
import ToolsSection from "./ToolsSection";
import Pricing from "../Payment/Pricing";  // fixed import
import CTASection from "./CTASection";
import Footer from "./Footer";
import Navbar from "./Navbar";

export default function LandingPage() {
  return (
    <div className="bg-white text-gray-900 min-h-screen">
      <Navbar />
      <Hero />
      <Features />
      <ToolsSection />
      <Pricing />
      <CTASection />
      <Footer />
    </div>
  );
}
