import { useMutation, UseMutationResult } from "@tanstack/react-query";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import useAuthorized from "../useAuthorized";

export interface StoreModelInDBParams {
  store_model_in_db: boolean;
}

export interface StoreModelInDBResponse {
  message: string;
}

const performStoreModelInDB = async (
  accessToken: string,
  params: StoreModelInDBParams
): Promise<StoreModelInDBResponse> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/config/field/update` : `/config/field/update`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      field_name: "store_model_in_db",
      field_value: params.store_model_in_db,
      config_type: "general_settings",
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage =
      errorData?.error?.message || errorData?.message || errorData?.detail || "Failed to update model storage settings";
    throw new Error(errorMessage);
  }

  const data = await response.json();
  return data;
};

export const useStoreModelInDB = (): UseMutationResult<
  StoreModelInDBResponse,
  Error,
  StoreModelInDBParams
> => {
  const { accessToken } = useAuthorized();

  return useMutation<StoreModelInDBResponse, Error, StoreModelInDBParams>({
    mutationFn: async (params: StoreModelInDBParams) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await performStoreModelInDB(accessToken, params);
    },
  });
};
