import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { useMutation } from "@tanstack/react-query";
import { LOCAL_CLOUDZERO_ERROR } from "./cloudZeroErrors";

interface CreateParams {
  connection_id: string;
  timezone?: string;
  api_key?: string;
}

interface CreateResponse {
  [key: string]: any;
}

const performCloudZeroCreate = async (accessToken: string, params: CreateParams): Promise<CreateResponse> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/cloudzero/init` : `/cloudzero/init`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      connection_id: params.connection_id,
      timezone: params.timezone ?? "UTC",
      ...(params.api_key && { api_key: params.api_key }),
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage = errorData?.error?.message || errorData?.message || errorData?.detail;
    throw errorMessage
      ? new Error(errorMessage)
      : new Error("Failed to create CloudZero integration", { cause: LOCAL_CLOUDZERO_ERROR });
  }

  const data = await response.json();
  return data;
};

export const useCloudZeroCreate = (accessToken: string) => {
  return useMutation<CreateResponse, Error, CreateParams>({
    mutationFn: async (params: CreateParams) => {
      if (!accessToken) {
        throw new Error("Access token is required", { cause: LOCAL_CLOUDZERO_ERROR });
      }
      return await performCloudZeroCreate(accessToken, params);
    },
  });
};
