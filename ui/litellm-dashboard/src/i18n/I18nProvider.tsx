"use client";

import LanguageProvider from "./LanguageProvider";

export default function I18nProvider({ children }: { children: React.ReactNode }) {
  return <LanguageProvider>{children}</LanguageProvider>;
}
