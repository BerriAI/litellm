import { CloudZeroSettings } from "@/components/CloudZeroCostTracking/types";
import { getProxyBaseUrl } from "@/components/networking";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const cloudZeroSettingsKeys = createQueryKeys("cloudZeroSettings");

const getCloudZeroSettings = async (accessToken: string): Promise<CloudZeroSettings | null> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/cloudzero/settings` : `/cloudzero/settings`;

  const response = await fetch(url, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });

  if (response.status === 404) {
    // 404 means no settings are configured - this is expected and not an error
    return null;
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage =
      errorData?.error?.message || errorData?.message || errorData?.detail || "Failed to fetch CloudZero settings";
    throw new Error(errorMessage);
  }

  const data = await response.json();
  return data;
};

export const useCloudZeroSettings = (accessToken: string) => {
  return useQuery<CloudZeroSettings | null>({
    queryKey: cloudZeroSettingsKeys.list({}),
    queryFn: async () => await getCloudZeroSettings(accessToken),
    enabled: !!accessToken && !!getProxyBaseUrl(),
    staleTime: 60 * 60 * 1000, // 1 hour - data rarely changes
    gcTime: 60 * 60 * 1000, // 1 hour - keep in cache for 1 hour
  });
};

interface UpdateParams {
  connection_id?: string;
  timezone?: string;
  api_key?: string;
}

interface UpdateResponse {
  message: string;
  status: string;
}

interface DeleteResponse {
  message: string;
  status: string;
}

const updateCloudZeroSettings = async (accessToken: string, params: UpdateParams): Promise<UpdateResponse> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/cloudzero/settings` : `/cloudzero/settings`;

  const response = await fetch(url, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...(params.connection_id && { connection_id: params.connection_id }),
      ...(params.timezone && { timezone: params.timezone }),
      ...(params.api_key && { api_key: params.api_key }),
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage =
      errorData?.error?.message || errorData?.message || errorData?.detail || "Failed to update CloudZero settings";
    throw new Error(errorMessage);
  }

  const data = await response.json();
  return data;
};

export const useCloudZeroUpdateSettings = (accessToken: string) => {
  const queryClient = useQueryClient();

  return useMutation<UpdateResponse, Error, UpdateParams>({
    mutationFn: async (params: UpdateParams) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await updateCloudZeroSettings(accessToken, params);
    },
    onSuccess: () => {
      // Invalidate the settings query to refetch updated data
      queryClient.invalidateQueries({ queryKey: cloudZeroSettingsKeys.list({}) });
    },
  });
};

const deleteCloudZeroSettings = async (accessToken: string): Promise<DeleteResponse> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/cloudzero/delete` : `/cloudzero/delete`;

  const response = await fetch(url, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage =
      errorData?.error?.message || errorData?.message || errorData?.detail || "Failed to delete CloudZero settings";
    throw new Error(errorMessage);
  }

  const data = await response.json();
  return data;
};

export const useCloudZeroDeleteSettings = (accessToken: string) => {
  const queryClient = useQueryClient();

  return useMutation<DeleteResponse, Error, void>({
    mutationFn: async () => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await deleteCloudZeroSettings(accessToken);
    },
    onSuccess: () => {
      // Invalidate the settings query to refetch updated data
      queryClient.invalidateQueries({ queryKey: cloudZeroSettingsKeys.list({}) });
    },
  });
};
