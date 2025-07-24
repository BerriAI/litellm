import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { getProxyBaseUrl } from '@/components/networking';

interface ThemeColors {
  brand_color_primary: string;
  brand_color_muted: string;
  brand_color_subtle: string;
  brand_color_faint: string;
  brand_color_emphasis: string;
}

interface ThemeContextType {
  colors: ThemeColors;
  updateColors: (newColors: Partial<ThemeColors>) => void;
  resetColors: () => void;
  logoUpdateTrigger: number;
  triggerLogoUpdate: () => void;
  logoUrl: string | null;
  setLogoUrl: (url: string | null) => void;
}

const defaultColors: ThemeColors = {
  brand_color_primary: '#6366f1',
  brand_color_muted: '#8688ef',
  brand_color_subtle: '#8e91eb',
  brand_color_faint: '#e0e7ff',
  brand_color_emphasis: '#5558eb',
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: ReactNode;
  accessToken?: string | null;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children, accessToken }) => {
  const [colors, setColors] = useState<ThemeColors>(defaultColors);
  const [logoUpdateTrigger, setLogoUpdateTrigger] = useState(0);
  const [logoUrl, setLogoUrl] = useState<string | null>(null);

  // Function to convert hex to RGB
  const hexToRgb = (hex: string): string => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return '99, 102, 241'; // fallback to default indigo
    const r = parseInt(result[1], 16);
    const g = parseInt(result[2], 16);
    const b = parseInt(result[3], 16);
    return `${r}, ${g}, ${b}`;
  };

  // Function to generate lighter/darker variants
  const adjustColorBrightness = (hex: string, factor: number): string => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return hex;
    
    let r = parseInt(result[1], 16);
    let g = parseInt(result[2], 16);
    let b = parseInt(result[3], 16);
    
    r = Math.round(Math.min(255, Math.max(0, r + (factor * 255))));
    g = Math.round(Math.min(255, Math.max(0, g + (factor * 255))));
    b = Math.round(Math.min(255, Math.max(0, b + (factor * 255))));
    
    const toHex = (n: number) => {
      const hex = n.toString(16);
      return hex.length === 1 ? '0' + hex : hex;
    };
    
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  };

  // Apply theme colors to CSS variables
  const applyThemeColors = (themeColors: ThemeColors) => {
    const root = document.documentElement;
    
    // Set CSS variables for custom properties
    root.style.setProperty('--brand-primary', themeColors.brand_color_primary);
    root.style.setProperty('--brand-muted', themeColors.brand_color_muted);
    root.style.setProperty('--brand-subtle', themeColors.brand_color_subtle);
    root.style.setProperty('--brand-faint', themeColors.brand_color_faint);
    root.style.setProperty('--brand-emphasis', themeColors.brand_color_emphasis);

    // Set Tremor-compatible CSS variables (RGB format)
    root.style.setProperty('--tremor-brand-default', hexToRgb(themeColors.brand_color_primary));
    root.style.setProperty('--tremor-brand-muted', hexToRgb(themeColors.brand_color_muted));
    root.style.setProperty('--tremor-brand-subtle', hexToRgb(themeColors.brand_color_subtle));
    root.style.setProperty('--tremor-brand-faint', hexToRgb(themeColors.brand_color_faint));
    root.style.setProperty('--tremor-brand-emphasis', hexToRgb(themeColors.brand_color_emphasis));

    // Generate additional variants for comprehensive theming
    const primary = themeColors.brand_color_primary;
    root.style.setProperty('--brand-50', adjustColorBrightness(primary, 0.95));
    root.style.setProperty('--brand-100', adjustColorBrightness(primary, 0.9));
    root.style.setProperty('--brand-200', adjustColorBrightness(primary, 0.8));
    root.style.setProperty('--brand-300', adjustColorBrightness(primary, 0.6));
    root.style.setProperty('--brand-400', adjustColorBrightness(primary, 0.3));
    root.style.setProperty('--brand-500', primary);
    root.style.setProperty('--brand-600', adjustColorBrightness(primary, -0.1));
    root.style.setProperty('--brand-700', adjustColorBrightness(primary, -0.2));
    root.style.setProperty('--brand-800', adjustColorBrightness(primary, -0.3));
    root.style.setProperty('--brand-900', adjustColorBrightness(primary, -0.4));
  };

  // Load theme from backend or localStorage on mount
  useEffect(() => {
    const loadThemeSettings = async () => {
      // First try to load from backend if we have an access token
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
            if (data.values) {
              // Extract color values from the response
              const themeColors = {
                brand_color_primary: data.values.brand_color_primary || defaultColors.brand_color_primary,
                brand_color_muted: data.values.brand_color_muted || defaultColors.brand_color_muted,
                brand_color_subtle: data.values.brand_color_subtle || defaultColors.brand_color_subtle,
                brand_color_faint: data.values.brand_color_faint || defaultColors.brand_color_faint,
                brand_color_emphasis: data.values.brand_color_emphasis || defaultColors.brand_color_emphasis,
              };
              setColors(themeColors);
              // Also save to localStorage for faster subsequent loads
              localStorage.setItem('litellm-theme-colors', JSON.stringify(themeColors));
              
              // Also set logo URL if present
              if (data.values.logo_url) {
                setLogoUrl(data.values.logo_url);
              }
              return;
            }
          }
        } catch (error) {
          console.warn('Failed to load theme settings from backend:', error);
        }
      }
      
      // Fall back to localStorage if backend fetch fails or no access token
      const savedTheme = localStorage.getItem('litellm-theme-colors');
      if (savedTheme) {
        try {
          const parsed = JSON.parse(savedTheme);
          setColors(parsed);
        } catch (error) {
          console.warn('Failed to parse saved theme colors:', error);
        }
      }
    };
    
    loadThemeSettings();
  }, [accessToken]);

  // Apply colors whenever they change
  useEffect(() => {
    applyThemeColors(colors);
  }, [colors]);

  const updateColors = (newColors: Partial<ThemeColors>) => {
    const updatedColors = { ...colors, ...newColors };
    setColors(updatedColors);
    
    // Save to localStorage for persistence
    localStorage.setItem('litellm-theme-colors', JSON.stringify(updatedColors));
  };

  const resetColors = () => {
    setColors(defaultColors);
    localStorage.removeItem('litellm-theme-colors');
  };

  const triggerLogoUpdate = () => {
    setLogoUpdateTrigger(prev => prev + 1);
  };

  return (
    <ThemeContext.Provider value={{ colors, updateColors, resetColors, logoUpdateTrigger, triggerLogoUpdate, logoUrl, setLogoUrl }}>
      {children}
    </ThemeContext.Provider>
  );
}; 