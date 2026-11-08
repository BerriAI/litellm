import type { ParseKeys, TFunction } from "i18next";

interface OptionItem<V extends string> {
  value: V;
  labelKey: ParseKeys;
}

export const ACTION_ITEMS: readonly OptionItem<"BLOCK" | "MASK">[] = [
  { value: "BLOCK", labelKey: "guardrails.contentFilter.options.action.block" },
  { value: "MASK", labelKey: "guardrails.contentFilter.options.action.mask" },
];

export const SEVERITY_ITEMS: readonly OptionItem<"high" | "medium" | "low">[] = [
  { value: "high", labelKey: "guardrails.contentFilter.options.severity.high" },
  { value: "medium", labelKey: "guardrails.contentFilter.options.severity.medium" },
  { value: "low", labelKey: "guardrails.contentFilter.options.severity.low" },
];

export const translateOptions = <V extends string>(items: readonly OptionItem<V>[], t: TFunction) =>
  items.map(({ value, labelKey }) => ({ value, label: t(labelKey) }));
