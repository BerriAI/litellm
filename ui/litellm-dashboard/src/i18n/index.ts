import { createInstance, type InitOptions } from "i18next";
import en from "./locales/en.json";
import zhCN from "./locales/zh-CN.json";

export type Language = "en" | "zh-CN";
export const LANGUAGE_STORAGE_KEY = "litellm_ui_language";

export function isLanguage(value: unknown): value is Language {
  return value === "en" || value === "zh-CN";
}

export function createDashboardI18n(language: Language = "en") {
  const instance = createInstance();
  const options: InitOptions = {
    lng: language,
    fallbackLng: "en",
    supportedLngs: ["en", "zh-CN"],
    resources: { en: { translation: { ...en } }, "zh-CN": { translation: { ...zhCN } } },
    initAsync: false,
    keySeparator: false,
    nsSeparator: false,
    interpolation: { escapeValue: false },
    react: { useSuspense: false },
  };
  void instance.init(options);
  return instance;
}
