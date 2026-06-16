"use client";

import React from "react";
import { Button } from "antd";
import { useLanguage } from "@/contexts/LanguageContext";

/**
 * Inline language toggle used in navbar.
 * Shows the opposite language (中文 ↔ English).
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useLanguage();

  const toggle = () => setLocale(locale === "en" ? "zh" : "en");

  return (
    <Button
      type="text"
      size="small"
      onClick={toggle}
      className="text-gray-600 hover:text-gray-900"
      title={t("language.label")}
    >
      {t("language.switch_to")}
    </Button>
  );
}

/**
 * Wraps children in LanguageProvider.
 * This should be placed above ConfigProvider in the component tree.
 */
export function LanguageProviderWrapper({ children }: { children: React.ReactNode }) {
  return <LanguageProvider>{children}</LanguageProvider>;
}

export { LanguageProvider };
