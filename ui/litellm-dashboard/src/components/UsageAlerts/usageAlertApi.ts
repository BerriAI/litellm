import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { backgroundFetchClient } from "@/lib/http/api";
import type { paths } from "@/lib/http/schema";
import { getProxyBaseUrl } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";
import { uiHref } from "@/utils/uiHref";

export type AlertPage = paths["/usage/alerts"]["get"]["responses"][200]["content"]["application/json"];
export type UsageAlert = AlertPage["data"][number];
export type RulePage = paths["/usage/alert-rules"]["get"]["responses"][200]["content"]["application/json"];
export type UsageRule = RulePage["data"][number];
export type RuleInput = paths["/usage/alert-rules"]["post"]["requestBody"]["content"]["application/json"];
export type AlertStatus = UsageAlert["status"];
export const METRIC_LABELS = { spend: "Spend", requests: "Requests", latency: "Average latency", errors: "Errors" };
export const SCOPE_LABELS = { deployment: "Deployment", key: "Key", team: "Team", user: "User", model: "Model" };

export const usageAlertHref = (alert: UsageAlert) => uiHref(alert.activity_path);
export const usageAlertEntity = (alert: UsageAlert) =>
  alert.scope_id ? `${SCOPE_LABELS[alert.scope_type]}: ${alert.scope_id}` : SCOPE_LABELS[alert.scope_type];

export function formatAlertValue(metric: UsageAlert["metric"], value: number): string {
  if (metric === "spend")
    return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 4 }).format(
      value,
    );
  if (metric === "latency") return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

export function useUsageAlerts(accessToken: string | null, status: AlertStatus = "open", offset = 0, limit = 50) {
  const { data: user } = useCurrentUser(accessToken);
  const enabled = Boolean(accessToken && user?.user_id && hasCapability(user.user_role, "viewGlobalSpend"));
  const options: UseQueryOptions<AlertPage> = {
    queryKey: ["usageAlerts", getProxyBaseUrl(), user?.user_id, user?.user_role, status, offset, limit],
    enabled,
    queryFn: async ({ signal }) => {
      const { data } = await backgroundFetchClient.GET("/usage/alerts", {
        params: { query: { status, offset, limit } },
        signal,
      });
      if (!data) throw new Error("Could not load usage alerts");
      return data;
    },
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
    retry: false,
  };
  const query = useQuery(options);
  return { ...query, data: enabled ? query.data : undefined };
}

export async function updateAlertStatus(id: string, status: AlertStatus) {
  const { data } = await backgroundFetchClient.PATCH("/usage/alerts/{alert_id}", {
    params: { path: { alert_id: id } },
    body: { status },
  });
  if (!data) throw new Error("Could not update alert");
  return data;
}
