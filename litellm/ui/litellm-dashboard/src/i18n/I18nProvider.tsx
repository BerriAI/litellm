"use client";

import type { i18n as I18n } from "i18next";
import { useEffect, useState, useSyncExternalStore } from "react";
import { I18nextProvider } from "react-i18next";

import LoadingScreen from "@/components/common_components/LoadingScreen";

import i18n from "./i18n";
import { browserLanguageStore, DEFAULT_LANGUAGE, LanguageStore, Locales, LOCALES, resolveLanguage } from "./languages";

type LanguageGateProps = {
  i18n: I18n;
  locales: Locales;
  store: LanguageStore;
  children: React.ReactNode;
};

const neverChanges = () => () => undefined;

export function LanguageGate({ i18n, locales, store, children }: LanguageGateProps) {
  const language = useSyncExternalStore(
    neverChanges,
    () => resolveLanguage(store.read(), locales),
    () => DEFAULT_LANGUAGE,
  );
  const [settled, setSettled] = useState(DEFAULT_LANGUAGE);

  useEffect(() => {
    const load = locales.get(language);
    if (!load) return;
    load()
      .then((messages) => {
        i18n.addResourceBundle(language, "translation", messages);
        return i18n.changeLanguage(language);
      })
      .then(() => {
        document.documentElement.lang = language;
      })
      .catch((error: unknown) => console.warn(`Couldn't load the "${language}" translations`, error))
      .finally(() => setSettled(language));
  }, [i18n, language, locales]);

  return language === settled ? children : <LoadingScreen />;
}

export default function I18nProvider({ children }: { children: React.ReactNode }) {
  return (
    <I18nextProvider i18n={i18n}>
      <LanguageGate i18n={i18n} locales={LOCALES} store={browserLanguageStore}>
        {children}
      </LanguageGate>
    </I18nextProvider>
  );
}
