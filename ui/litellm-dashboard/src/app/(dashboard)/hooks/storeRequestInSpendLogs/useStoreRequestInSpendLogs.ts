import { useMutation, UseMutationResult } from "@tanstack/react-query";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import useAuthorized from "../useAuthorized";

export interface StoreRequestInSpendLogsParams {
  store_prompts_in_spend_logs: boolean;
  maximum_spend_logs_retention_period?: string;
}

export interface StoreRequestInSpendLogsResponse {
  message: string;
}

const performStoreRequestInSpendLogs = async (
  accessToken: string,
  params: StoreRequestInSpendLogsParams
): Promise<StoreRequestInSpendLogsResponse> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/config/update` : `/config/update`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      general_settings: {
        store_prompts_in_spend_logs: params.store_prompts_in_spend_logs,
        ...(params.maximum_spend_logs_retention_period && {
          maximum_spend_logs_retention_period: params.maximum_spend_logs_retention_period,
        }),
      },
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage =
      errorData?.error?.message || errorData?.message || errorData?.detail || "Failed to update spend logs settings";
    throw new Error(errorMessage);
  }

  const data = await response.json();
  return data;
};

export const useStoreRequestInSpendLogs = (): UseMutationResult<
  StoreRequestInSpendLogsResponse,
  Error,
  StoreRequestInSpendLogsParams
> => {
  const { accessToken } = useAuthorized();

  return useMutation<StoreRequestInSpendLogsResponse, Error, StoreRequestInSpendLogsParams>({
    mutationFn: async (params: StoreRequestInSpendLogsParams) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await performStoreRequestInSpendLogs(accessToken, params);
    },
  });
};
