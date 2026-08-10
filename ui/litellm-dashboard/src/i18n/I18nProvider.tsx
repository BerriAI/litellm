"use client";

import LoadingScreen from "@/components/common_components/LoadingScreen";
import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";
import { createInstance } from "i18next";
import { createContext, PropsWithChildren, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { I18nextProvider, initReactI18next } from "react-i18next";
import { LANGUAGE_STORAGE_KEY, resolveLanguage, SupportedLanguage } from "./language";
import { resources } from "./resources";

interface DashboardLanguageContextValue {
  language: SupportedLanguage;
  setLanguage: (language: SupportedLanguage) => Promise<void>;
}

const DashboardLanguageContext = createContext<DashboardLanguageContextValue | null>(null);

export const I18nProvider = ({ children }: PropsWithChildren) => {
  const [i18n] = useState(() => createInstance().use(initReactI18next));
  const [language, setActiveLanguage] = useState<SupportedLanguage | null>(null);

  useEffect(() => {
    let active = true;
    const initialLanguage = resolveLanguage(getLocalStorageItem(LANGUAGE_STORAGE_KEY), navigator.language);
    const initializationOptions = {
      resources,
      lng: initialLanguage,
      fallbackLng: "en",
      interpolation: { escapeValue: false },
    };

    void i18n.init(initializationOptions).then(() => {
      if (!active) return;
      document.documentElement.lang = initialLanguage;
      setActiveLanguage(initialLanguage);
    });

    return () => {
      active = false;
    };
  }, [i18n]);

  const setLanguage = useCallback(
    async (nextLanguage: SupportedLanguage) => {
      await i18n.changeLanguage(nextLanguage);
      setLocalStorageItem(LANGUAGE_STORAGE_KEY, nextLanguage);
      document.documentElement.lang = nextLanguage;
      setActiveLanguage(nextLanguage);
    },
    [i18n],
  );

  const value = useMemo(() => (language === null ? null : { language, setLanguage }), [language, setLanguage]);

  if (value === null) return <LoadingScreen />;

  return (
    <I18nextProvider i18n={i18n}>
      <DashboardLanguageContext.Provider value={value}>{children}</DashboardLanguageContext.Provider>
    </I18nextProvider>
  );
};

export const useDashboardLanguage = (): DashboardLanguageContextValue => {
  const value = useContext(DashboardLanguageContext);
  if (value === null) throw new Error("useDashboardLanguage must be used within I18nProvider");
  return value;
};
