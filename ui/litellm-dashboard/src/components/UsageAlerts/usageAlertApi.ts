import { useQuery } from "@tanstack/react-query";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { backgroundFetchClient } from "@/lib/http/api";
import type { paths } from "@/lib/http/schema";
import { getProxyBaseUrl } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";
import { uiHref } from "@/utils/uiHref";

type AlertPage = paths["/usage/alerts"]["get"]["responses"][200]["content"]["application/json"];
export type UsageAlert = AlertPage["data"][number];
export type AlertGroupPage = paths["/usage/alert-groups"]["get"]["responses"][200]["content"]["application/json"];
export type AlertGroup = AlertGroupPage["data"][number];
export type AlertModelsPage = paths["/usage/alert-models"]["get"]["responses"][200]["content"]["application/json"];
export type RulePage = paths["/usage/alert-rules"]["get"]["responses"][200]["content"]["application/json"];
export type UsageRule = RulePage["data"][number];
export type RuleInput = paths["/usage/alert-rules"]["post"]["requestBody"]["content"]["application/json"];
export type AlertStatus = UsageAlert["status"];
export const METRIC_LABELS = { spend: "Spend", requests: "Requests", latency: "Average latency", errors: "Errors" };
export const METRIC_SHORT_LABELS = { spend: "Cost", requests: "Requests", latency: "Latency", errors: "Errors" };
export const SCOPE_LABELS = { deployment: "Gateway", key: "Key", team: "Team", user: "User", model: "Model" };

export const usageAlertHref = (alert: UsageAlert) => uiHref(alert.activity_path);
export const usageAlertEntity = (alert: UsageAlert) =>
  alert.scope_id ? `${SCOPE_LABELS[alert.scope_type]}: ${alert.scope_id}` : SCOPE_LABELS[alert.scope_type];

export const usageAlertGroupHref = (group: AlertGroup) => uiHref(group.activity_path);

export function usageAlertGroupName(group: AlertGroup): string {
  if (group.scope_label) return group.scope_label;
  if (!group.scope_id) return SCOPE_LABELS[group.scope_type];
  if (group.scope_type === "model") return group.scope_id;
  const id = group.scope_id.length > 24 ? `${group.scope_id.slice(0, 8)}…${group.scope_id.slice(-6)}` : group.scope_id;
  return `${SCOPE_LABELS[group.scope_type]} ${id}`;
}

export function usageAlertGroupTitle(group: AlertGroup): string {
  if (group.scope_label && group.scope_id) return `${group.scope_label} (${group.scope_id})`;
  if (group.scope_label) return group.scope_label;
  if (group.scope_id) return `${SCOPE_LABELS[group.scope_type]} ${group.scope_id}`;
  return SCOPE_LABELS[group.scope_type];
}

export function formatAlertChange(alert: Pick<UsageAlert, "baseline_value" | "observed_value" | "direction">): string {
  if (alert.baseline_value === 0) return alert.observed_value > 0 ? "New activity" : "No change";
  if (alert.observed_value === 0) return "↓ 100%";
  const ratio =
    alert.direction === "increase"
      ? alert.observed_value / alert.baseline_value
      : alert.baseline_value / alert.observed_value;
  const formatted = ratio.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return `${alert.direction === "increase" ? "↑" : "↓"} ${formatted}×`;
}

export function formatAlertValue(metric: UsageAlert["metric"], value: number): string {
  if (metric === "spend")
    return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 4 }).format(
      value,
    );
  if (metric === "latency") return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

export function useUsageAlertGroups(
  accessToken: string | null,
  status: AlertStatus = "open",
  options: { offset?: number; limit?: number; scope?: UsageAlert["scope_type"] } = {},
) {
  const { offset = 0, limit = 25, scope } = options;
  const { data: user } = useCurrentUser(accessToken);
  const enabled = Boolean(accessToken && user?.user_id && hasCapability(user.user_role, "viewGlobalSpend"));
  const queryOptions = {
    queryKey: ["usageAlertGroups", getProxyBaseUrl(), user?.user_id, user?.user_role, status, offset, limit, scope],
    enabled,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/usage/alert-groups", {
        params: { query: { status, offset, limit, scope_type: scope } },
        signal,
      });
      if (!data) throw new Error("Could not load performance alerts");
      return data;
    },
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
    retry: false,
  };
  const query = useQuery(queryOptions);
  return { ...query, data: enabled ? query.data : undefined };
}

export async function updateAlertGroup(group: AlertGroup, expectedStatus: AlertStatus, status: AlertStatus) {
  const { data } = await backgroundFetchClient.PATCH("/usage/alert-groups", {
    body: {
      scope_type: group.scope_type,
      scope_id: group.scope_id,
      day: group.day,
      expected_status: expectedStatus,
      status,
    },
  });
  if (!data) throw new Error("Could not update the alerts");
  return data;
}

export async function updateAlertStatus(id: string, status: AlertStatus) {
  const { data } = await backgroundFetchClient.PATCH("/usage/alerts/{alert_id}", {
    params: { path: { alert_id: id } },
    body: { status },
  });
  if (!data) throw new Error("Could not update alert");
  return data;
}
