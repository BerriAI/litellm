import type { components } from "@/lib/http/schema";
import type { AlertThresholdsFields } from "./schema";

type Thresholds = components["schemas"]["PerformanceAlertThresholds"];
export const thresholdRules = ["spend", "traffic", "latency", "reliability"] as const;
export type ThresholdRule = (typeof thresholdRules)[number];

const minimumScale = (rule: ThresholdRule) => (rule === "reliability" ? 100 : 1);

export const displayedMinimum = (thresholds: Thresholds, rule: ThresholdRule) =>
  Number((thresholds[rule].minimum_increase * minimumScale(rule)).toPrecision(15));

export const toAlertThresholdsFields = (thresholds: Thresholds): AlertThresholdsFields =>
  Object.fromEntries(
    thresholdRules.map((rule) => [
      rule,
      { multiplier: String(thresholds[rule].multiplier), minimum: String(displayedMinimum(thresholds, rule)) },
    ]),
  ) as AlertThresholdsFields;

export const toThresholds = (fields: AlertThresholdsFields): Thresholds =>
  Object.fromEntries(
    thresholdRules.map((rule) => [
      rule,
      {
        multiplier: Number(fields[rule].multiplier),
        minimum_increase: Number(fields[rule].minimum) / minimumScale(rule),
      },
    ]),
  ) as Thresholds;
