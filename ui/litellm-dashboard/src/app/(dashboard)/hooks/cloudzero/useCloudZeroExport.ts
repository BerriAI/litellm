import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { useMutation } from "@tanstack/react-query";

interface ExportParams {
  operation?: string;
}

interface ExportResponse {
  [key: string]: any;
}

const performCloudZeroExport = async (accessToken: string, params: ExportParams = {}): Promise<ExportResponse> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/cloudzero/export` : `/cloudzero/export`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      operation: params.operation ?? "replace_hourly",
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage =
      errorData?.error?.message || errorData?.message || errorData?.detail || "Failed to export data";
    throw new Error(errorMessage);
  }

  const data = await response.json();
  return data;
};

export const useCloudZeroExport = (accessToken: string) => {
  return useMutation<ExportResponse, Error, ExportParams>({
    mutationFn: async (params: ExportParams = {}) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await performCloudZeroExport(accessToken, params);
    },
  });
};
