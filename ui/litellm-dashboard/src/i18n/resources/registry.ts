import type { ResourceLanguage } from "i18next";

import commonEn from "@/locales/en/common.json";
import navigationEn from "@/locales/en/navigation.json";
import authEn from "@/locales/en/auth.json";
import modelsEn from "@/locales/en/models.json";
import apiKeysEn from "@/locales/en/apiKeys.json";
import usageEn from "@/locales/en/usage.json";
import costEn from "@/locales/en/cost.json";
import budgetsEn from "@/locales/en/budgets.json";

import commonZh from "@/locales/zh-CN/common.json";
import navigationZh from "@/locales/zh-CN/navigation.json";
import authZh from "@/locales/zh-CN/auth.json";
import modelsZh from "@/locales/zh-CN/models.json";
import apiKeysZh from "@/locales/zh-CN/apiKeys.json";
import usageZh from "@/locales/zh-CN/usage.json";
import costZh from "@/locales/zh-CN/cost.json";
import budgetsZh from "@/locales/zh-CN/budgets.json";

export const supportedLngs = ["en", "zh-CN"] as const;
export type Locale = (typeof supportedLngs)[number];

/**
 * Namespace registry — the single source of truth for namespaces and their
 * static resources. Agent 4 owns this file permanently (FILE_OWNERSHIP rule 6).
 * Function agents only add keys to their own JSON; any new namespace must be
 * routed through Agent 4.
 */
// Namespace registry — the single source of truth for namespaces and their
// static resources. Agent 4 owns this file permanently (FILE_OWNERSHIP rule 6).
// Function agents only add keys to their own JSON; any new namespace must be
// routed through Agent 4.
//
// KEY TYPING NOTE: use `satisfies Record<Namespace, object>` — NOT
// `Record<Namespace, ResourceKey>`. The broad `ResourceKey` target would erase
// each namespace's concrete key shapes, so `CustomTypeOptions.resources`
// (types.d.ts) could only type the defaultNS (common) keys and could not infer
// qualified keys like `navigation:dashboard`. `satisfies object` only validates
// the shape without widening, keeping the `as const` literal keys intact.
export const NAMESPACES = [
  "common",
  "navigation",
  "auth",
  "models",
  "apiKeys",
  "usage",
  "cost",
  "budgets",
] as const;
export type Namespace = (typeof NAMESPACES)[number];

const enResources = {
  common: commonEn,
  navigation: navigationEn,
  auth: authEn,
  models: modelsEn,
  apiKeys: apiKeysEn,
  usage: usageEn,
  cost: costEn,
  budgets: budgetsEn,
} as const satisfies Record<Namespace, object>;

const zhCNResources = {
  common: commonZh,
  navigation: navigationZh,
  auth: authZh,
  models: modelsZh,
  apiKeys: apiKeysZh,
  usage: usageZh,
  cost: costZh,
  budgets: budgetsZh,
} as const satisfies Record<Namespace, object>;

/**
 * Resource loading map: locale -> namespace -> static JSON.
 * English is the type source of truth (D2). Kept as a literal (`as const`) so
 * `typeof RESOURCES["en"]` drives `CustomTypeOptions['resources']` with concrete
 * key shapes for typed `t()` keys.
 */
export const RESOURCES = {
  en: enResources,
  "zh-CN": zhCNResources,
} as const satisfies Record<Locale, ResourceLanguage>;

export const defaultNS: Namespace = "common";
