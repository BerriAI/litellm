import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";

export type Messages = { readonly [key: string]: string | Messages };
export type Locales = ReadonlyMap<string, () => Promise<Messages>>;
export type LanguageStore = { readonly read: () => string | null; readonly write: (language: string) => void };

export const DEFAULT_LANGUAGE = "en";
export const LANGUAGE_STORAGE_KEY = "litellm_language";

export const LOCALES: Locales = new Map();

export const browserLanguageStore: LanguageStore = {
  read: () => getLocalStorageItem(LANGUAGE_STORAGE_KEY),
  write: (language) => setLocalStorageItem(LANGUAGE_STORAGE_KEY, language),
};

export const resolveLanguage = (
  stored: string | null,
  locales: Locales,
  browserLanguages: readonly string[] = [],
): string => {
  if (stored !== null && locales.has(stored)) return stored;

  const supported = [DEFAULT_LANGUAGE, ...locales.keys()];
  for (const candidate of browserLanguages) {
    const subtags = candidate.trim().replace(/_/g, "-").toLowerCase().split("-").filter(Boolean);
    // Try the full browser locale first, then progressively less-specific tags
    // (for example, fr-CA -> fr) when only a base translation is available.
    for (let length = subtags.length; length > 0; length -= 1) {
      const tag = subtags.slice(0, length).join("-");
      const match = supported.find((language) => language.toLowerCase() === tag);
      if (match) return match;
    }
  }

  return DEFAULT_LANGUAGE;
};

export const supportedLanguages = (locales: Locales): readonly string[] => [DEFAULT_LANGUAGE, ...locales.keys()];

const displayName = (language: string, displayLanguage: string): string =>
  new Intl.DisplayNames([displayLanguage], { type: "language" }).of(language) ?? language;

export const nativeName = (language: string): string => {
  const name = displayName(language, language);
  return name.charAt(0).toLocaleUpperCase(language) + name.slice(1);
};

export const englishName = (language: string): string => displayName(language, DEFAULT_LANGUAGE);
