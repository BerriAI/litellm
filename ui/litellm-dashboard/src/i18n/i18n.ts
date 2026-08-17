import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./locales/en";
import zhCN from "./locales/zh-CN";

export const supportedLanguages = ["en", "zh-CN"] as const;
export type SupportedLanguage = (typeof supportedLanguages)[number];
export const languageStorageKey = "litellm_ui_language";

export const normalizeLanguage = (language: string | null | undefined): SupportedLanguage =>
  language?.toLowerCase().startsWith("zh") ? "zh-CN" : "en";

export const detectInitialLanguage = (
  storedLanguage: string | null,
  browserLanguages: readonly string[],
): SupportedLanguage => {
  if (storedLanguage && supportedLanguages.includes(storedLanguage as SupportedLanguage)) {
    return storedLanguage as SupportedLanguage;
  }
  return normalizeLanguage(browserLanguages.find((language) => language.toLowerCase().startsWith("zh")));
};

if (!i18n.isInitialized) {
  void i18n.use(initReactI18next).init({
    resources: {
      en: { translation: en },
      "zh-CN": { translation: zhCN },
    },
    lng: "en",
    fallbackLng: "en",
    supportedLngs: supportedLanguages,
    interpolation: { escapeValue: false },
    initAsync: false,
  });
}

export default i18n;
