import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from "react";
import { ConfigProvider, theme as antdTheme } from "antd";
import { getProxyBaseUrl } from "@/components/networking";

const DARK_MODE_STORAGE_KEY = "litellm-dark-mode";

interface ThemeContextType {
  logoUrl: string | null;
  setLogoUrl: (url: string | null) => void;
  faviconUrl: string | null;
  setFaviconUrl: (url: string | null) => void;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  setIsDarkMode: (value: boolean) => void;
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

const readInitialDarkMode = (): boolean => {
  if (typeof window === "undefined") return false;
  try {
    const stored = window.localStorage.getItem(DARK_MODE_STORAGE_KEY);
    if (stored !== null) return stored === "true";
  } catch {
    // ignore localStorage access errors (e.g. privacy mode)
  }
  return false;
};

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children, accessToken }) => {
  const [logoUrl, setLogoUrl] = useState<string | null>(null);
  const [faviconUrl, setFaviconUrl] = useState<string | null>(null);
  const [isDarkMode, setIsDarkModeState] = useState<boolean>(false);
  const [hydrated, setHydrated] = useState(false);

  // Hydrate state from the value the inline init script already applied
  // (see `darkModeInitScript` in app/layout.tsx). Until this runs we leave
  // the DOM untouched so the SSR'd class set by the script survives.
  useEffect(() => {
    setIsDarkModeState(readInitialDarkMode());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated || typeof document === "undefined") return;
    const root = document.documentElement;
    if (isDarkMode) {
      root.classList.add("dark");
      root.style.colorScheme = "dark";
    } else {
      root.classList.remove("dark");
      root.style.colorScheme = "light";
    }
    try {
      window.localStorage.setItem(DARK_MODE_STORAGE_KEY, String(isDarkMode));
    } catch {
      // ignore localStorage write errors
    }
  }, [isDarkMode, hydrated]);

  const setIsDarkMode = useCallback((value: boolean) => {
    setIsDarkModeState(value);
  }, []);

  const toggleDarkMode = useCallback(() => {
    setIsDarkModeState((prev) => !prev);
  }, []);

  useEffect(() => {
    const loadThemeSettings = async () => {
      try {
        const proxyBaseUrl = getProxyBaseUrl();
        const url = proxyBaseUrl ? `${proxyBaseUrl}/get/ui_theme_settings` : "/get/ui_theme_settings";
        const response = await fetch(url, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });

        if (response.ok) {
          const data = await response.json();
          if (data.values?.logo_url) {
            setLogoUrl(data.values.logo_url);
          }
          if (data.values?.favicon_url) {
            setFaviconUrl(data.values.favicon_url);
          }
        }
      } catch (error) {
        console.warn("Failed to load theme settings from backend:", error);
      }
    };

    loadThemeSettings();
  }, []);

  useEffect(() => {
    if (faviconUrl) {
      const existingLinks = document.querySelectorAll("link[rel*='icon']");
      if (existingLinks.length > 0) {
        existingLinks.forEach((link) => {
          (link as HTMLLinkElement).href = faviconUrl;
        });
      } else {
        const link = document.createElement("link");
        link.rel = "icon";
        link.href = faviconUrl;
        document.head.appendChild(link);
      }
    }
  }, [faviconUrl]);

  return (
    <ThemeContext.Provider
      value={{ logoUrl, setLogoUrl, faviconUrl, setFaviconUrl, isDarkMode, toggleDarkMode, setIsDarkMode }}
    >
      <ConfigProvider
        theme={{
          algorithm: isDarkMode ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        }}
      >
        {children}
      </ConfigProvider>
    </ThemeContext.Provider>
  );
};
