import i18n from "@/i18n/i18n";
import type { TrendMetric } from "./usageTrendData";

export const formatExactUsd = (value: number): string =>
  i18n.t("format.usd", {
    val: value,
    minimumFractionDigits: 2,
    maximumFractionDigits: value !== 0 && Math.abs(value) < 1 ? 4 : 2,
  });

const formatCompactUsd = (value: number): string => {
  if (Math.abs(value) >= 1000) {
    return i18n.t("format.usd", { val: value, notation: "compact", maximumFractionDigits: 1 });
  }
  if (value !== 0 && Math.abs(value) < 1) return i18n.t("format.usd", { val: value, maximumSignificantDigits: 2 });
  return i18n.t("format.usd", { val: value, minimumFractionDigits: 0, maximumFractionDigits: 2 });
};

export const formatExactCount = (value: number): string => i18n.t("format.number", { val: value });

const formatCompactCount = (value: number): string => i18n.t("format.compactNumber", { val: value });

export const EXACT_FORMAT: Readonly<Record<TrendMetric, (value: number) => string>> = {
  spend: formatExactUsd,
  tokens: formatExactCount,
  requests: formatExactCount,
};

export const COMPACT_FORMAT: Readonly<Record<TrendMetric, (value: number) => string>> = {
  spend: formatCompactUsd,
  tokens: formatCompactCount,
  requests: formatCompactCount,
};
