import dayjs from "dayjs";
import "dayjs/locale/zh-cn";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "@/locales/en.json";
import zhCN from "@/locales/zh-CN.json";
import { getLocalStorageItem } from "@/utils/localStorageUtils";

export const LANGUAGE_STORAGE_KEY = "litellm_ui_language";

export const SUPPORTED_LANGUAGES = [
  { code: "en", label: "English" },
  { code: "zh-CN", label: "简体中文" },
] as const;

type LanguageCode = (typeof SUPPORTED_LANGUAGES)[number]["code"];

const DAYJS_LOCALES: Record<string, string> = { "zh-CN": "zh-cn" };

export function detectLanguage(): LanguageCode {
  if (typeof window === "undefined") return "en";

  const stored = getLocalStorageItem(LANGUAGE_STORAGE_KEY);
  const fromStorage = SUPPORTED_LANGUAGES.find((lang) => lang.code === stored);
  if (fromStorage) return fromStorage.code;

  const browser = window.navigator.language;
  const exact = SUPPORTED_LANGUAGES.find((lang) => lang.code === browser);
  if (exact) return exact.code;

  const primarySubtag = browser?.split("-")[0];
  const byPrimarySubtag = SUPPORTED_LANGUAGES.find((lang) => lang.code.split("-")[0] === primarySubtag);
  return byPrimarySubtag?.code ?? "en";
}

// Registered before init so the initial language also syncs these side effects
i18n.on("languageChanged", (lng) => {
  dayjs.locale(DAYJS_LOCALES[lng] ?? "en");
  if (typeof window !== "undefined") {
    document.documentElement.lang = lng;
  }
});

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    "zh-CN": { translation: zhCN },
  },
  lng: detectLanguage(),
  fallbackLng: "en",
  // React already escapes interpolated values
  interpolation: { escapeValue: false },
  // Initialize synchronously so SSG prerendering and tests never render untranslated keys
  initAsync: false,
});

export default i18n;
