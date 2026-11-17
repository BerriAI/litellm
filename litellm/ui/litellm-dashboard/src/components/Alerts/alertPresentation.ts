import type { components } from "@/lib/http/schema";
import i18n from "@/i18n/i18n";
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
export const TYPE_LABEL_KEYS = {
  all: "alerts.unified.types.all",
  performance: "alerts.unified.types.performance",
  "model-deprecations": "alerts.page.tabs.modelDeprecations",
  spend: "alerts.unified.types.spend",
  traffic: "alerts.unified.types.traffic",
  latency: "alerts.unified.types.latency",
  reliability: "alerts.unified.types.reliability",
} as const;
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
      title: i18n.t(
        alert.model.status === "deprecated" ? "alerts.unified.retiredTitle" : "alerts.unified.retiresTitle",
        {
          name: alert.usageModel,
          date: dateLabel(alert.model.deprecation_date),
        },
      ),
      description: i18n.t("alerts.unified.deprecationDescription", {
        requests: alert.requests.toLocaleString(),
        date: dateLabel(alert.lastUsedDate),
      }),
      status: effectiveAlertStatus(alert),
      resourceType: "model" as const,
      resourceName: alert.usageModel,
      href: modelGroupHref(alert.model.model_name),
      linkLabel: i18n.t("alerts.unified.links.model"),
      unit: "requests" as MetricUnit,
      history: alert.history ?? null,
      baseline: undefined,
      observed: alert.lastUsedDate,
      dates: [
        [
          i18n.t(alert.model.status === "deprecated" ? "alerts.unified.retired" : "alerts.unified.retires"),
          alert.model.deprecation_date,
        ],
        [i18n.t("alerts.unified.lastUsed"), alert.lastUsedDate],
      ],
      chartTitle: i18n.t("alerts.usageChart.title"),
    };
  }
  const alert = item.alert;
  const name =
    alert.resource_name ||
    (alert.resource_type === "key"
      ? i18n.t("alerts.unified.keyFallback", { id: alert.resource_id.slice(0, 8) })
      : alert.resource_id);
  if (!("value" in alert.evidence)) {
    return {
      id: alert.id,
      title: i18n.t("alerts.unified.retiresTitle", { name, date: dateLabel(alert.evidence.deprecation_date) }),
      description: i18n.t("alerts.unified.deprecation"),
      status: alert.status,
      resourceType: alert.resource_type,
      resourceName: name,
      href: modelGroupHref(alert.resource_id),
      linkLabel: i18n.t("alerts.unified.links.model"),
      unit: "requests" as MetricUnit,
      history: null,
      baseline: undefined,
      observed: alert.window_start.slice(0, 10),
      dates: [[i18n.t("alerts.unified.retires"), alert.evidence.deprecation_date]],
      chartTitle: i18n.t("alerts.usageChart.title"),
    };
  }
  const evidence = alert.evidence;
  const observed = alert.window_start.slice(0, 10);
  const label = i18n.t(`alerts.unified.metrics.${alert.type}`);
  const direction = evidence.value < evidence.baseline ? "fell" : "rose";
  return {
    id: alert.id,
    title:
      evidence.baseline > 0
        ? i18n.t(`alerts.unified.comparison.${direction}`, {
            label,
            name,
            ratio: (evidence.value / evidence.baseline).toLocaleString(undefined, { maximumFractionDigits: 2 }),
          })
        : i18n.t(`alerts.unified.value.${direction}`, {
            label,
            name,
            value: metricValue(evidence.value, evidence.unit),
          }),
    description: i18n.t("alerts.unified.description", {
      value: metricValue(evidence.value, evidence.unit),
      date: dateLabel(observed),
      baseline: metricValue(evidence.baseline, evidence.unit),
    }),
    status: alert.status,
    resourceType: alert.resource_type,
    resourceName: name,
    href: activityLink(alert),
    linkLabel: i18n.t(`alerts.unified.links.${alert.resource_type}`),
    unit: evidence.unit,
    history: alert.history,
    baseline: evidence.baseline,
    observed,
    dates: [
      [i18n.t("alerts.unified.observed"), observed],
      [i18n.t("alerts.unified.firstDetected"), alert.created_at],
    ],
    chartTitle: i18n.t(`alerts.unified.chart.titles.${evidence.unit}`),
  };
}
export type AlertPresentation = ReturnType<typeof presentAlert>;
