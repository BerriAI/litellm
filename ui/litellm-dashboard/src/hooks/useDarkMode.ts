import { useEffect, useState } from "react";

export const DARK_MODE_STORAGE_KEY = "litellm-dark-mode";

export const readStoredDarkMode = (): boolean => {
  if (typeof window === "undefined") return false;
  const stored = localStorage.getItem(DARK_MODE_STORAGE_KEY);
  if (stored === null) return window.matchMedia("(prefers-color-scheme: dark)").matches;
  return stored === "true";
};

export const useDarkMode = (): { isDarkMode: boolean; toggleDarkMode: () => void } => {
  const [isDarkMode, setIsDarkMode] = useState<boolean>(readStoredDarkMode);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDarkMode);
  }, [isDarkMode]);

  const toggleDarkMode = () => {
    setIsDarkMode((prev) => {
      const next = !prev;
      if (typeof window !== "undefined") {
        localStorage.setItem(DARK_MODE_STORAGE_KEY, String(next));
      }
      return next;
    });
  };

  return { isDarkMode, toggleDarkMode };
};
