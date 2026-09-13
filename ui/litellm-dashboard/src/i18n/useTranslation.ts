"use client";

import { useContext } from "react";
import { I18nContext, useTranslation as useReactTranslation } from "react-i18next";
import { createDashboardI18n } from ".";

const englishFallback = createDashboardI18n();

export function useTranslation() {
  const context = useContext(I18nContext);
  return useReactTranslation(undefined, { i18n: context?.i18n ?? englishFallback });
}
