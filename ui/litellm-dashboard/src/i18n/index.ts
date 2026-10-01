"use client";

// ./i18n.ts imports react-i18next, whose module scope calls React context APIs.
// Without this directive, importing the barrel from a server component (e.g.
// app/layout.tsx) evaluates that module in the server graph and the build fails
// with "createContext is not a function".
export { I18nProvider, useI18n } from "./I18nProvider";
export { getI18n } from "./i18n";
export type { Locale } from "./resources/registry";
export { supportedLngs, NAMESPACES } from "./resources/registry";
