import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { getGuardrailsList } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

// ── Types ────────────────────────────────────────────────────────────────────

export interface GuardrailListItem {
  guardrail_name: string;
  litellm_params?: {
    default_on?: boolean;
    mode?: string | string[];
    [key: string]: unknown;
  };
  guardrail_info?: Record<string, unknown> | null;
  guardrail_id?: string | null;
  [key: string]: unknown;
}

interface GuardrailsListResponse {
  guardrails: GuardrailListItem[];
}

export interface GuardrailsListData {
  guardrails: GuardrailListItem[];
  globalGuardrailNames: Set<string>;
  optionalGuardrailNames: Set<string>;
}

// ── Hook ─────────────────────────────────────────────────────────────────────

const guardrailKeys = createQueryKeys("guardrails");

export const useGuardrails = (): UseQueryResult<GuardrailsListData> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<GuardrailsListResponse, Error, GuardrailsListData>({
    queryKey: guardrailKeys.list({}),
    queryFn: async () => getGuardrailsList(accessToken!),
    enabled: Boolean(accessToken && userId && userRole),
    select: (data) => {
      const guardrails: GuardrailListItem[] = data?.guardrails ?? [];
      const globalGuardrailNames = new Set<string>();
      const optionalGuardrailNames = new Set<string>();
      for (const g of guardrails) {
        if (g.litellm_params?.default_on) {
          globalGuardrailNames.add(g.guardrail_name);
        } else {
          optionalGuardrailNames.add(g.guardrail_name);
        }
      }
      return { guardrails, globalGuardrailNames, optionalGuardrailNames };
    },
  });
};
