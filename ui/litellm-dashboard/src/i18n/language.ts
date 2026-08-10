export const SUPPORTED_LANGUAGES = ["en", "ru"] as const;

export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

export const LANGUAGE_STORAGE_KEY = "litellm_ui_language";

const isSupportedLanguage = (value: string | null): value is SupportedLanguage =>
  SUPPORTED_LANGUAGES.some((language) => language === value);

export const resolveLanguage = (saved: string | null, browserLanguage: string): SupportedLanguage => {
  if (isSupportedLanguage(saved)) return saved;
  return browserLanguage.toLowerCase().startsWith("ru") ? "ru" : "en";
};
