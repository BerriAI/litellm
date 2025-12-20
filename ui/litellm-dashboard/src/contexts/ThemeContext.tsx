import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { getProxyBaseUrl } from "@/components/networking";

type ThemeMode = "light" | "dark" | "system";

interface ThemeContextType {
  logoUrl: string | null;
  setLogoUrl: (url: string | null) => void;
  themeMode: ThemeMode;
  setThemeMode: (mode: ThemeMode) => void;
  isDarkMode: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
};

interface ThemeProviderProps {
  children: ReactNode;
  accessToken?: string | null;
}

const THEME_STORAGE_KEY = "litellm-theme-mode";

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children, accessToken }) => {
  const [logoUrl, setLogoUrl] = useState<string | null>(null);
  const [themeMode, setThemeModeState] = useState<ThemeMode>("system");
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Initialize theme from localStorage
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null;
      if (savedTheme && ["light", "dark", "system"].includes(savedTheme)) {
        setThemeModeState(savedTheme);
      }
    }
  }, []);

  // Update isDarkMode based on themeMode and system preference
  useEffect(() => {
    if (typeof window === "undefined") return;

    const updateDarkMode = () => {
      let shouldBeDark = false;
      if (themeMode === "dark") {
        shouldBeDark = true;
      } else if (themeMode === "system") {
        shouldBeDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      }
      setIsDarkMode(shouldBeDark);

      // Apply dark class to html element
      if (shouldBeDark) {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    };

    updateDarkMode();

    // Listen for system preference changes
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => {
      if (themeMode === "system") {
        updateDarkMode();
      }
    };
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [themeMode]);

  const setThemeMode = (mode: ThemeMode) => {
    setThemeModeState(mode);
    if (typeof window !== "undefined") {
      localStorage.setItem(THEME_STORAGE_KEY, mode);
    }
  };

  // Load logo URL from backend on mount
  // Note: /get/ui_theme_settings is now a public endpoint (no auth required)
  // so all users can see custom branding set by admins
  useEffect(() => {
    const loadLogoSettings = async () => {
      try {
        const proxyBaseUrl = getProxyBaseUrl();
        const url = proxyBaseUrl ? `${proxyBaseUrl}/get/ui_theme_settings` : "/get/ui_theme_settings";
        const response = await fetch(url, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (response.ok) {
          const data = await response.json();
          if (data.values?.logo_url) {
            setLogoUrl(data.values.logo_url);
          }
        }
      } catch (error) {
        console.warn("Failed to load logo settings from backend:", error);
      }
    };

    loadLogoSettings();
  }, []);

  return (
    <ThemeContext.Provider value={{ logoUrl, setLogoUrl, themeMode, setThemeMode, isDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
};
