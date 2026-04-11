import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { createQueryKeys } from "../common/queryKeysFactory";

// ── Types ────────────────────────────────────────────────────────────────────

export interface RegisterGuardrailParams {
  guardrail_name: string;
  litellm_params: Record<string, unknown>;
  guardrail_info?: Record<string, unknown>;
  team_id?: string;
}

export interface RegisterGuardrailResponse {
  guardrail_id: string;
  guardrail_name: string;
  status: string;
  submitted_at?: string | null;
}

// ── Fetch function ───────────────────────────────────────────────────────────

const registerGuardrail = async (
  accessToken: string,
  params: RegisterGuardrailParams,
): Promise<RegisterGuardrailResponse> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/guardrails/register`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage = deriveErrorMessage(errorData);
    handleError(errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
};

// ── Hook ─────────────────────────────────────────────────────────────────────

const guardrailKeys = createQueryKeys("guardrails");

export const useRegisterGuardrail = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<RegisterGuardrailResponse, Error, RegisterGuardrailParams>({
    mutationFn: async (params) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return registerGuardrail(accessToken, params);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: guardrailKeys.all });
    },
  });
};
