import { useQuery, useMutation, UseMutationResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import useAuthorized from "../useAuthorized";
import { proxyBaseUrl, getGlobalLitellmHeaderName, deriveErrorMessage, handleError } from "@/components/networking";

/**
 * Enum for config types that can be fetched from the proxy config endpoint.
 * Currently supports general_settings, but can be extended as more config types are added.
 */
export enum ConfigType {
  GENERAL_SETTINGS = "general_settings",
}

/**
 * Enum for supported field names that can be deleted from general_settings.
 * This should match the fields available in ConfigGeneralSettings.
 */
export enum GeneralSettingsFieldName {
  MAXIMUM_SPEND_LOGS_RETENTION_PERIOD = "maximum_spend_logs_retention_period",
  // Add more field names here as needed
}

/**
 * Field detail for nested fields within a config field
 */
export interface FieldDetail {
  field_name: string;
  field_type: string;
  field_description: string;
  field_default_value: any;
  stored_in_db: boolean | null;
}

/**
 * Configuration list item returned from /config/list endpoint
 */
export interface ConfigListItem {
  field_name: string;
  field_type: string;
  field_description: string;
  field_value: any;
  stored_in_db: boolean | null;
  field_default_value: any;
  premium_field?: boolean;
  nested_fields?: FieldDetail[] | null;
}

/**
 * Response type for /config/list endpoint
 */
export type ProxyConfigResponse = ConfigListItem[];

/**
 * Request body for /config/field/delete endpoint
 */
export interface DeleteProxyConfigFieldRequest {
  config_type: ConfigType;
  field_name: string;
}

/**
 * Response type for /config/field/delete endpoint
 */
export interface DeleteProxyConfigFieldResponse {
  message?: string;
  [key: string]: any;
}

/**
 * Network call function to fetch proxy config by config type
 * @param accessToken - The access token for authentication
 * @param configType - The type of config to fetch (from ConfigType enum)
 * @returns Promise resolving to the config list response
 */
export const getProxyConfigCall = async (accessToken: string, configType: ConfigType): Promise<ProxyConfigResponse> => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/list?config_type=${configType}`
      : `/config/list?config_type=${configType}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to get proxy config for ${configType}:`, error);
    throw error;
  }
};

const proxyConfigKeys = createQueryKeys("proxyConfig");

/**
 * Network call function to delete a proxy config field
 * @param accessToken - The access token for authentication
 * @param request - The delete request containing config_type and field_name
 * @returns Promise resolving to the delete response
 */
export const deleteProxyConfigFieldCall = async (
  accessToken: string,
  request: DeleteProxyConfigFieldRequest,
): Promise<DeleteProxyConfigFieldResponse> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/config/field/delete` : `/config/field/delete`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to delete proxy config field ${request.field_name}:`, error);
    throw error;
  }
};

/**
 * React Query hook to fetch proxy config by config type
 * @param configType - The type of config to fetch (from ConfigType enum)
 * @returns React Query result with the config list data
 */
export const useProxyConfig = (configType: ConfigType) => {
  const { accessToken } = useAuthorized();
  return useQuery<ProxyConfigResponse>({
    queryKey: proxyConfigKeys.list({
      filters: {
        configType,
      },
    }),
    queryFn: async () => await getProxyConfigCall(accessToken!, configType),
    enabled: Boolean(accessToken),
  });
};

/**
 * React Query hook to delete a proxy config field
 * @returns React Query mutation result for deleting config fields
 */
export const useDeleteProxyConfigField = (): UseMutationResult<
  DeleteProxyConfigFieldResponse,
  Error,
  DeleteProxyConfigFieldRequest
> => {
  const { accessToken } = useAuthorized();

  return useMutation<DeleteProxyConfigFieldResponse, Error, DeleteProxyConfigFieldRequest>({
    mutationFn: async (request: DeleteProxyConfigFieldRequest) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await deleteProxyConfigFieldCall(accessToken, request);
    },
  });
};
