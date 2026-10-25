import { englishTranslate, type Translate } from "@/i18n/language";
import type { ReminderMarkerPair } from "./build_complexity_router_config";
import type { ClassifierType } from "./classifier_types";

// 本地输入校验接受当前语言，默认英文以保持现有纯函数调用兼容。
export const getHeuristicV2SuccessThresholdError = (
  threshold: number | undefined,
  t: Translate = englishTranslate,
): string | null => {
  if (threshold === undefined) return null;
  const validProbability = Number.isFinite(threshold) && threshold >= 0 && threshold <= 1;
  return validProbability ? null : t("Success threshold must be a number between 0 and 1");
};

export const getReminderMarkersError = (
  pairs: ReminderMarkerPair[] | undefined,
  t: Translate = englishTranslate,
): string | null => {
  for (const [index, pair] of (pairs ?? []).entries()) {
    const open = pair.open.trim().toLowerCase();
    const close = pair.close.trim().toLowerCase();
    if (!open || !close) return t("Tag pair {index} needs both an opening and a closing tag", { index: index + 1 });
    if (open === close) return t("Tag pair {index} must use different opening and closing tags", { index: index + 1 });
  }
  return null;
};

export const getClassifierPluginTimeoutError = (
  classifierType: ClassifierType,
  timeoutMs: number | undefined,
  t: Translate = englishTranslate,
): string | null => {
  if (classifierType !== "custom" || timeoutMs === undefined) return null;
  return Number.isInteger(timeoutMs) && timeoutMs > 0
    ? null
    : t("Classifier plugin timeout must be a whole number of milliseconds greater than 0");
};
