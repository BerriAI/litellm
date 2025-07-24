import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { getProxyBaseUrl } from '@/components/networking'

interface ThemeContextType {
  logoUrl: string | null;
  setLogoUrl: (url: string | null) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

interface ThemeProviderProps {
  children: ReactNode;
  accessToken?: string | null;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children, accessToken }) => {
  const [logoUrl, setLogoUrl] = useState<string | null>(null);

  // Load logo URL from backend on mount
  useEffect(() => {
    const loadLogoSettings = async () => {
      if (accessToken) {
        try {
          const proxyBaseUrl = getProxyBaseUrl();
          const url = proxyBaseUrl ? `${proxyBaseUrl}/get/ui_theme_settings` : '/get/ui_theme_settings';
          const response = await fetch(url, {
            method: 'GET',
            headers: {
              'Authorization': `Bearer ${accessToken}`,
              'Content-Type': 'application/json',
            },
          });
          
          if (response.ok) {
            const data = await response.json();
            if (data.values?.logo_url) {
              setLogoUrl(data.values.logo_url);
            }
          }
        } catch (error) {
          console.warn('Failed to load logo settings from backend:', error);
        }
      }
    };
    
    loadLogoSettings();
  }, [accessToken]);

  return (
    <ThemeContext.Provider value={{ logoUrl, setLogoUrl }}>
      {children}
    </ThemeContext.Provider>
  );
};