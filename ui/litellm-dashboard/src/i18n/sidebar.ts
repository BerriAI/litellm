import { SupportedLanguage } from "./language";
import { resources } from "./resources";

export const getSidebarTranslations = (language: SupportedLanguage) => resources[language].navigation.sidebar;
