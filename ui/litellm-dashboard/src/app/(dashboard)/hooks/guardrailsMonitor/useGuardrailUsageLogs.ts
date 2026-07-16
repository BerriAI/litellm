import { $api } from "@/lib/http/api";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { components } from "@/lib/http/schema";

export type GuardrailUsageLogs = components["schemas"]["UsageLogsResponse"];
export type GuardrailUsageLogEntry = components["schemas"]["UsageLogEntry"];
export type GuardrailLogAction = GuardrailUsageLogEntry["action"];

interface UseGuardrailUsageLogsParams {
  guardrailId: string;
  page: number;
  pageSize: number;
  startDate: string;
  endDate: string;
}

export const useGuardrailUsageLogs = ({
  guardrailId,
  page,
  pageSize,
  startDate,
  endDate,
}: UseGuardrailUsageLogsParams) => {
  const { accessToken } = useAuthorized();

  return $api.useQuery(
    "get",
    "/guardrails/usage/logs",
    {
      params: {
        query: {
          guardrail_id: guardrailId,
          page,
          page_size: pageSize,
          start_date: startDate,
          end_date: endDate,
        },
      },
    },
    { enabled: Boolean(accessToken) && Boolean(guardrailId) },
  );
};
