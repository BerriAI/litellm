import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { getProxyBaseUrl } from "@/components/networking";

interface ThemeContextType {
  logoUrl: string | null;
  setLogoUrl: (url: string | null) => void;
  faviconUrl: string | null;
  setFaviconUrl: (url: string | null) => void;
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

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children, accessToken }) => {
  const [logoUrl, setLogoUrl] = useState<string | null>(null);
  const [faviconUrl, setFaviconUrl] = useState<string | null>(null);

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
    <ThemeContext.Provider value={{ logoUrl, setLogoUrl, faviconUrl, setFaviconUrl }}>
      {children}
    </ThemeContext.Provider>
  );
};
