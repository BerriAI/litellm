import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

export type GuardrailUsageOverview = components["schemas"]["UsageOverviewResponse"];
export type GuardrailUsageOverviewRow = components["schemas"]["UsageOverviewRow"];
export type GuardrailUsageDetail = components["schemas"]["UsageDetailResponse"];

export interface GuardrailsUsageWindow {
  accessToken: string | null;
  startDate: string;
  endDate: string;
}

const dateQuery = (startDate: string, endDate: string) => ({
  start_date: startDate || undefined,
  end_date: endDate || undefined,
});

export const useGuardrailsUsageOverview = ({ accessToken, startDate, endDate }: GuardrailsUsageWindow) =>
  $api.useQuery(
    "get",
    "/guardrails/usage/overview",
    { params: { query: dateQuery(startDate, endDate) } },
    { enabled: Boolean(accessToken) },
  );

export const useGuardrailsUsageDetail = (
  guardrailId: string,
  { accessToken, startDate, endDate }: GuardrailsUsageWindow,
) =>
  $api.useQuery(
    "get",
    "/guardrails/usage/detail/{guardrail_id}",
    { params: { path: { guardrail_id: guardrailId }, query: dateQuery(startDate, endDate) } },
    { enabled: Boolean(accessToken && guardrailId) },
  );
