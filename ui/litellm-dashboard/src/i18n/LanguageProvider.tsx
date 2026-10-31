"use client";

import { useEffect, useState, type ReactNode } from "react";
import { I18nextProvider } from "react-i18next";
import { createDashboardI18n, isLanguage, LANGUAGE_STORAGE_KEY } from ".";

export default function LanguageProvider({ children }: { children: ReactNode }) {
  const [instance] = useState(createDashboardI18n);

  useEffect(() => {
    const updateDocumentLanguage = (language: string) => {
      document.documentElement.lang = language;
    };
    instance.on("languageChanged", updateDocumentLanguage);
    try {
      const stored = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
      if (isLanguage(stored)) void instance.changeLanguage(stored);
    } catch {}
    updateDocumentLanguage(instance.language);
    return () => {
      instance.off("languageChanged", updateDocumentLanguage);
    };
  }, [instance]);

  return <I18nextProvider i18n={instance}>{children}</I18nextProvider>;
}
