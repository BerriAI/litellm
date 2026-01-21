import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { useMutation } from "@tanstack/react-query";

interface DryRunParams {
  limit?: number;
}

interface DryRunResponse {
  [key: string]: any;
}

const performCloudZeroDryRun = async (accessToken: string, params: DryRunParams = {}): Promise<DryRunResponse> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/cloudzero/dry-run` : `/cloudzero/dry-run`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      limit: params.limit ?? 10,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage =
      errorData?.error?.message || errorData?.message || errorData?.detail || "Failed to perform dry run";
    throw new Error(errorMessage);
  }

  const data = await response.json();
  return data;
};

export const useCloudZeroDryRun = (accessToken: string) => {
  return useMutation<DryRunResponse, Error, DryRunParams>({
    mutationFn: async (params: DryRunParams = {}) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await performCloudZeroDryRun(accessToken, params);
    },
  });
};
