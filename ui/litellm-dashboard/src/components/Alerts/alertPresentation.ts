import type { components } from "@/lib/http/schema";
import type { DeprecationAlert } from "@/components/ModelDeprecationAlerts/deprecationAlerts";
import { effectiveAlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { keyDetailHref, teamDetailHref, userDetailHref, modelGroupHref } from "@/utils/entityLinks";
import { uiHref } from "@/utils/uiHref";

export type SavedAlert = components["schemas"]["AlertView"];
export type AlertItem =
  | { source: "performance"; alert: SavedAlert }
  | { source: "deprecation"; alert: DeprecationAlert };
export type MetricUnit = "usd" | "requests" | "milliseconds" | "ratio";
export type HistoryPoint = { date: string; value: number | null };
export const ALERT_TYPES = [
  "all",
  "performance",
  "model-deprecations",
  "spend",
  "traffic",
  "latency",
  "reliability",
] as const;
export type AlertFilter = (typeof ALERT_TYPES)[number];
export const TYPE_LABELS: Record<AlertFilter, string> = {
  all: "All alerts",
  performance: "Performance",
  "model-deprecations": "Model deprecations",
  spend: "Spend",
  traffic: "Traffic",
  latency: "Latency",
  reliability: "Reliability",
};
const dateFormat = { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" } as const;
export const dateLabel = (value: string) =>
  new Date(value.length === 10 ? `${value}T00:00:00Z` : value).toLocaleDateString(undefined, dateFormat);
export function metricValue(value: number, unit: MetricUnit): string {
  if (unit === "usd") return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" }).format(value);
  if (unit === "ratio") return `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
  if (unit === "milliseconds") return `${(value / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 })} s`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export function activityLink(alert: SavedAlert): string | undefined {
  const href = (() => {
    switch (alert.resource_type) {
      case "key":
        return keyDetailHref(alert.resource_id);
      case "team":
        return teamDetailHref(alert.resource_id);
      case "user":
        return userDetailHref(alert.resource_id);
      case "model":
        return modelGroupHref(alert.resource_id);
      case "gateway":
        return uiHref("usage");
    }
  })();
  if (!href || alert.resource_type === "model") return href;
  const from = new Date(Date.parse(alert.window_start) - 7 * 86400000).toISOString().slice(0, 10);
  const to = alert.window_start.slice(0, 10);
  const query =
    alert.resource_type === "gateway"
      ? `from=${from}&to=${to}`
      : `detail_tab=activity&activity_from=${from}&activity_to=${to}`;
  return `${href}${href.includes("?") ? "&" : "?"}${query}`;
}

export function presentAlert(item: AlertItem) {
  if (item.source === "deprecation") {
    const alert = item.alert;
    return {
      id: alert.id,
      title: `${alert.usageModel} ${alert.model.status === "deprecated" ? "retired" : "retires"} on ${dateLabel(alert.model.deprecation_date)}`,
      description: `${alert.requests.toLocaleString()} recorded requests. Last used ${dateLabel(alert.lastUsedDate)} UTC.`,
      status: effectiveAlertStatus(alert),
      resourceType: "model" as const,
      resourceName: alert.usageModel,
      href: modelGroupHref(alert.model.model_name),
      linkLabel: "View model routing",
      unit: "requests" as MetricUnit,
      history: alert.history ?? null,
      baseline: undefined,
      observed: alert.lastUsedDate,
      dates: [
        [alert.model.status === "deprecated" ? "Retired" : "Retires", alert.model.deprecation_date],
        ["Last used", alert.lastUsedDate],
      ],
      chartTitle: "Daily requests",
    };
  }
  const alert = item.alert;
  const name =
    alert.resource_name || (alert.resource_type === "key" ? `Key ${alert.resource_id.slice(0, 8)}` : alert.resource_id);
  if (!("value" in alert.evidence)) {
    return {
      id: alert.id,
      title: `${name} retires on ${dateLabel(alert.evidence.deprecation_date)}`,
      description: "Model deprecation",
      status: alert.status,
      resourceType: alert.resource_type,
      resourceName: name,
      href: modelGroupHref(alert.resource_id),
      linkLabel: "View model routing",
      unit: "requests" as MetricUnit,
      history: null,
      baseline: undefined,
      observed: alert.window_start.slice(0, 10),
      dates: [["Retires", alert.evidence.deprecation_date]],
      chartTitle: "Daily requests",
    };
  }
  const evidence = alert.evidence;
  const observed = alert.window_start.slice(0, 10);
  const label = {
    spend: "Spend",
    traffic: "Traffic",
    latency: "Mean response time",
    reliability: "Error rate",
    model_deprecation: "Usage",
  }[alert.type];
  const direction = evidence.value < evidence.baseline ? "fell" : "rose";
  const comparison =
    evidence.baseline > 0
      ? ` to ${(evidence.value / evidence.baseline).toLocaleString(undefined, { maximumFractionDigits: 2 })}× its baseline`
      : ` to ${metricValue(evidence.value, evidence.unit)}`;
  return {
    id: alert.id,
    title: `${label} on ${name} ${direction}${comparison}`,
    description: `${metricValue(evidence.value, evidence.unit)} on ${dateLabel(observed)} UTC, compared with the usual ${metricValue(evidence.baseline, evidence.unit)}.`,
    status: alert.status,
    resourceType: alert.resource_type,
    resourceName: name,
    href: activityLink(alert),
    linkLabel: {
      gateway: "View Usage",
      model: "View model routing",
      key: "View key activity",
      team: "View team activity",
      user: "View user activity",
    }[alert.resource_type],
    unit: evidence.unit,
    history: alert.history,
    baseline: evidence.baseline,
    observed,
    dates: [
      ["Observed", observed],
      ["First detected", alert.created_at],
    ],
    chartTitle: {
      usd: "Daily spend",
      requests: "Daily requests",
      milliseconds: "Mean response time per day",
      ratio: "Daily error rate",
    }[evidence.unit],
  };
}
export type AlertPresentation = ReturnType<typeof presentAlert>;
