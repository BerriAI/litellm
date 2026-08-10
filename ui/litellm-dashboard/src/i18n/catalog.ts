import { enAuth } from "./locales/en/auth";
import { enCommon } from "./locales/en/common";
import { enNavigation } from "./locales/en/navigation";
import { ruAuth } from "./locales/ru/auth";
import { ruCommon } from "./locales/ru/common";
import { ruNavigation } from "./locales/ru/navigation";

export const TRANSLATION_NAMESPACES = ["common", "auth", "navigation"] as const;

export type TranslationNamespace = (typeof TRANSLATION_NAMESPACES)[number];

export const resources = {
  en: { common: enCommon, auth: enAuth, navigation: enNavigation },
  ru: { common: ruCommon, auth: ruAuth, navigation: ruNavigation },
} as const;
