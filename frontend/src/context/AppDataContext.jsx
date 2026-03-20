// src/context/AppDataContext.jsx

import React, { createContext, useContext, useState, useEffect } from "react";

// Services
import { settingsService } from "../services/settingsService";
import { responseStyleService } from "../services/responseStyleService";
import { getUserPersonas } from "../database/personaService";

// Auth (FIXED PATH)
import { useAuth } from "../components/Auth/AuthContext";

// Create context
const AppDataContext = createContext();
export const useAppData = () => useContext(AppDataContext);

export function AppDataProvider({ children }) {
  const { user } = useAuth();

  // GLOBAL STATES
  const [loading, setLoading] = useState(true);
  const [settings, setSettings] = useState(null);
  const [styles, setStyles] = useState([]);
  const [personas, setPersonas] = useState([]);

  const [error, setError] = useState(null);

  // AUTO-LOAD WHEN USER LOGS IN
  useEffect(() => {
    if (user) {
      preloadAllData();
    } else {
      setSettings(null);
      setStyles([]);
      setPersonas([]);
      setLoading(false);
    }
  }, [user]);

  // =============================
  // 🔥 MASTER LOADER
  // =============================
  const preloadAllData = async () => {
    setLoading(true);
    setError(null);

    try {
      // 1️⃣ SETTINGS
      const userSettings = await settingsService.loadSettingsWithFallback(user.id);
      setSettings(userSettings);

      // 2️⃣ RESPONSE STYLES
      const styleList = await responseStyleService.getAllStyles(user.id);
      setStyles(styleList);

      // 3️⃣ PERSONAS
      const personaRes = await getUserPersonas();
      if (personaRes.success) {
        setPersonas(personaRes.data);
      } else {
        setError(personaRes.error || "Failed to load personas");
      }

    } catch (err) {
      console.error("GLOBAL PRELOAD ERROR:", err);
      setError("Failed to load app data");
    }

    setLoading(false);
  };

  return (
    <AppDataContext.Provider
      value={{
        loading,
        error,

        // GLOBAL DATA
        settings,
        styles,
        personas,

        // Reload function
        reloadAll: preloadAllData
      }}
    >
      {children}
    </AppDataContext.Provider>
  );
}
