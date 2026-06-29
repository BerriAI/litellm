/**
 * LiteLLM i18n - Lightweight translation system
 *
 * Supports English (default) and Chinese (中文).
 * Uses React Context for state + localStorage for persistence.
 * Zero external dependencies.
 */

import en from "./locales/en.json";
import zh from "./locales/zh.json";

export type Locale = "en" | "zh";
/** Supports arbitrarily nested JSON, not just flat two-level maps */
export type TranslationMap = Record<string, string | Record<string, string>>;

const translations: Record<Locale, TranslationMap> = { en, zh };

/**
 * Resolve a dotted key like "nav.dashboard" against the current locale.
 */
export function t(key: string, locale?: Locale): string {
  let resolvedLocale: Locale;
  if (locale) {
    resolvedLocale = locale;
  } else {
    try {
      const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
      resolvedLocale = (stored === "zh" || stored === "en") ? stored as Locale : DEFAULT_LOCALE;
    } catch {
      resolvedLocale = DEFAULT_LOCALE;
    }
  }

  const map = translations[resolvedLocale] || translations.en;
  const parts = key.split(".");
  let current: any = map;
  for (const part of parts) {
    if (current == null) break;
    current = current[part];
  }
  if (typeof current === "string") return current;

  // Fallback to English
  let fallback: any = translations.en;
  for (const part of parts) {
    if (fallback == null) break;
    fallback = fallback[part];
  }
  if (typeof fallback === "string") return fallback;

  return key;
}

export const SUPPORTED_LOCALES: { value: Locale; label: string }[] = [
  { value: "en", label: "English" },
  { value: "zh", label: "中文" },
];

export const LOCALE_STORAGE_KEY = "litellm_locale";
export const DEFAULT_LOCALE: Locale = "zh";
export { translations };
