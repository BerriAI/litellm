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
export type TranslationMap = Record<string, Record<string, string>>;

const translations: Record<Locale, TranslationMap> = { en, zh };

/**
 * Resolve a dotted key like "nav.dashboard" against the current locale.
 */
export function t(key: string, locale: Locale = "en"): string {
  const map = translations[locale] || translations.en;
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

  // Ultimate fallback: return the key itself
  return key;
}

/** All supported locales (used by the switcher) */
export const SUPPORTED_LOCALES: { value: Locale; label: string }[] = [
  { value: "en", label: "English" },
  { value: "zh", label: "中文" },
];

/** LocalStorage key */
export const LOCALE_STORAGE_KEY = "litellm_locale";

/** Default locale */
export const DEFAULT_LOCALE: Locale = "en";

export { translations };
