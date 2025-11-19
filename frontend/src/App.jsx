import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./components/Auth/AuthContext";
import ProtectedRoute from "./components/Auth/ProtectedRoute";

// Pages
import LandingPage from "./components/Landing/LandingPage";
import Interview from "./components/InterviewAssistent/Interview";
import InterviewAssist from "./components/InterviewAssist";
import Pricing from "./components/Payment/Pricing";
import PersonasPage from "./components/InterviewAssistent/PersonasPage";
import SignIn from "./components/Auth/SignIn";
import SignUp from "./components/Auth/SignUp";

// Resources
import Blog from "./components/Resources/Blog/Blog";
import ResumeBuilder from "./components/Resources/Blog/ResumeBuilder";   // ✅ Added

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>

          {/* Public Routes */}
          <Route path="/" element={<LandingPage />} />
          <Route path="/pricing" element={<Pricing />} />
          <Route path="/sign-in" element={<SignIn />} />
          <Route path="/sign-up" element={<SignUp />} />

          <Route path="/blog" element={<Blog />} />                    {/* ✅ Blog */}
          <Route path="/resume-builder" element={<ResumeBuilder />} /> {/* ✅ Resume Builder */}

          {/* Protected Routes */}
          <Route
            path="/interview"
            element={
              <ProtectedRoute>
                <Interview />
              </ProtectedRoute>
            }
          />

          <Route
            path="/interview-assist"
            element={
              <ProtectedRoute>
                <InterviewAssist />
              </ProtectedRoute>
            }
          />

          <Route
            path="/personas"
            element={
              <ProtectedRoute>
                <PersonasPage />
              </ProtectedRoute>
            }
          />

        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
