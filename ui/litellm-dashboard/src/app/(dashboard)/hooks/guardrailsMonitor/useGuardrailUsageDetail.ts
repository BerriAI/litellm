import { $api } from "@/lib/http/api";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { components } from "@/lib/http/schema";

export type GuardrailUsageDetail = components["schemas"]["UsageDetailResponse"];

export const useGuardrailUsageDetail = (guardrailId: string, startDate: string, endDate: string) => {
  const { accessToken } = useAuthorized();

  return $api.useQuery(
    "get",
    "/guardrails/usage/detail/{guardrail_id}",
    {
      params: {
        path: { guardrail_id: guardrailId },
        query: { start_date: startDate, end_date: endDate },
      },
    },
    { enabled: Boolean(accessToken) && Boolean(guardrailId) },
  );
};
