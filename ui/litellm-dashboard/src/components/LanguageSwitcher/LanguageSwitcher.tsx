"use client";

import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { useI18n, type Locale } from "@/i18n";

const LANGUAGE_OPTIONS: { value: Locale; key: string }[] = [
  // Keys are unprefixed: `common` is the default namespace (types.d.ts), so
  // i18next v26 typing resolves them without the `common:` prefix.
  { value: "en", key: "languages.en" },
  { value: "zh-CN", key: "languages.zh-CN" },
];

/**
 * EN / 中文 language toggle. Uses `useI18n().setLocale` which persists the
 * choice to the unified `litellm.locale` key and applies it to the active
 * i18next instance. The active option is the current locale.
 */
export function LanguageSwitcher() {
  const { locale, setLocale } = useI18n();
  const { t } = useTranslation();

  return (
    <div className="inline-flex items-center gap-1" role="group" aria-label={t("language.name")}>
      {LANGUAGE_OPTIONS.map((option) => {
        const isActive = option.value === locale;
        return (
          <Button
            key={option.value}
            type="button"
            size="sm"
            variant={isActive ? "secondary" : "ghost"}
            aria-pressed={isActive}
            onClick={() => setLocale(option.value)}
          >
            {t(option.key)}
          </Button>
        );
      })}
    </div>
  );
}
