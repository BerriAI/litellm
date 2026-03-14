import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { keyKeys } from "./useKeys";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ResetKeySpendResponse {
  key_hash: string;
  spend: number;
  previous_spend: number;
  max_budget: number | null;
  budget_reset_at: string | null;
}

// ── Fetch function ────────────────────────────────────────────────────────────

export const resetKeySpend = async (
  accessToken: string,
  keyToken: string,
): Promise<ResetKeySpendResponse> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl ? `${baseUrl}/key/${keyToken}/reset_spend` : `/key/${keyToken}/reset_spend`}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ reset_to: 0 }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    handleError(errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
};

// ── Hook ──────────────────────────────────────────────────────────────────────

export const useResetKeySpend = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<ResetKeySpendResponse, Error, string>({
    mutationFn: async (keyToken) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return resetKeySpend(accessToken, keyToken);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keyKeys.all });
    },
  });
};
