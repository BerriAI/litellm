"use client";

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { ConfigProvider } from "antd";
import enUS from "antd/locale/en_US";
import zhCN from "antd/locale/zh_CN";
import { Locale, DEFAULT_LOCALE, LOCALE_STORAGE_KEY, t as translate } from "@/i18n";

interface LanguageContextValue {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextValue>({
  locale: DEFAULT_LOCALE,
  setLocale: () => {},
  t: (key) => key,
});

export function useLanguage() {
  return useContext(LanguageContext);
}

const antdLocales: Record<Locale, any> = {
  en: enUS,
  zh: zhCN,
};

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(DEFAULT_LOCALE);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
      if (stored === "zh" || stored === "en") {
        setLocaleState(stored);
      }
    } catch {}
  }, []);

  // Sync <html lang> with locale (screen-reader detection)
  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const setLocale = useCallback((l: Locale) => {
    try { localStorage.setItem(LOCALE_STORAGE_KEY, l); } catch {}
    setLocaleState(l);
  }, []);

  const t = useCallback((key: string) => translate(key, locale), [locale]);

  return (
    <LanguageContext.Provider value={{ locale, setLocale, t }}>
      <ConfigProvider locale={antdLocales[locale]}>{children}</ConfigProvider>
    </LanguageContext.Provider>
  );
}
