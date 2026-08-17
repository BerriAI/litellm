"use client";

import type { PropsWithChildren } from "react";
import { useEffect } from "react";
import { I18nextProvider } from "react-i18next";
import i18n, { detectInitialLanguage, languageStorageKey, normalizeLanguage } from "./i18n";

export default function I18nProvider({ children }: PropsWithChildren) {
  useEffect(() => {
    const applyLanguage = (language: string) => {
      const normalizedLanguage = normalizeLanguage(language);
      document.documentElement.lang = normalizedLanguage;
      localStorage.setItem(languageStorageKey, normalizedLanguage);
    };
    const initialLanguage = detectInitialLanguage(localStorage.getItem(languageStorageKey), navigator.languages);

    i18n.on("languageChanged", applyLanguage);
    void i18n.changeLanguage(initialLanguage);
    applyLanguage(initialLanguage);

    return () => i18n.off("languageChanged", applyLanguage);
  }, []);

  return <I18nextProvider i18n={i18n}>{children}</I18nextProvider>;
}
