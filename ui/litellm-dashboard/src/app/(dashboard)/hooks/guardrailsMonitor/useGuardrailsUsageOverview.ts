import { $api } from "@/lib/http/api";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { components } from "@/lib/http/schema";

export type GuardrailUsageOverview = components["schemas"]["UsageOverviewResponse"];
export type GuardrailUsageRow = components["schemas"]["UsageOverviewRow"];
export type GuardrailUsageChartPoint = components["schemas"]["UsageChartPoint"];
export type GuardrailStatus = GuardrailUsageRow["status"];

export const useGuardrailsUsageOverview = (startDate: string, endDate: string) => {
  const { accessToken } = useAuthorized();

  return $api.useQuery(
    "get",
    "/guardrails/usage/overview",
    { params: { query: { start_date: startDate, end_date: endDate } } },
    { enabled: Boolean(accessToken) },
  );
};
