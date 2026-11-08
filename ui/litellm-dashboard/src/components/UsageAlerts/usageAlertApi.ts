import type { paths } from "@/lib/http/schema";
import { uiHref } from "@/utils/uiHref";

type AlertPage = paths["/alerts/signals"]["get"]["responses"][200]["content"]["application/json"];
export type PerformanceSignal = AlertPage["data"][number];
export type AlertModelsPage = paths["/alerts/models"]["get"]["responses"][200]["content"]["application/json"];
type RulePage = paths["/alerts/rules"]["get"]["responses"][200]["content"]["application/json"];
export type UsageRule = RulePage["data"][number];
export type RuleInput = paths["/alerts/rules"]["post"]["requestBody"]["content"]["application/json"];
export const METRIC_LABELS = {
  spend: "Spend",
  requests: "Requests",
  latency: "Average latency",
  errors: "Errors",
  error_rate: "Error rate",
};
export const SCOPE_LABELS = { deployment: "Gateway", key: "Key", team: "Team", user: "User", model: "Model" };

export const usageAlertHref = (alert: PerformanceSignal) => uiHref(alert.activity_path);

export function formatAlertChange(
  alert: Pick<PerformanceSignal, "baseline_value" | "observed_value" | "direction">,
): string {
  if (alert.baseline_value === 0) return alert.observed_value > 0 ? "New activity" : "No change";
  if (alert.observed_value === 0) return "↓ 100%";
  const ratio =
    alert.direction === "increase"
      ? alert.observed_value / alert.baseline_value
      : alert.baseline_value / alert.observed_value;
  const formatted = ratio.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return `${alert.direction === "increase" ? "↑" : "↓"} ${formatted}×`;
}

export function formatAlertValue(metric: PerformanceSignal["metric"], value: number): string {
  if (metric === "spend")
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: Math.abs(value) >= 0.01 ? 2 : 4,
    }).format(value);
  if (metric === "latency") return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms`;
  if (metric === "error_rate") return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}
