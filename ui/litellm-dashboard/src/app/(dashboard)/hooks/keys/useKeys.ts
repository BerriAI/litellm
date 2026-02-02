import { keepPreviousData, useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export const keyKeys = createQueryKeys("keys");

export interface KeysResponse {
  keys: KeyResponse[];
  total_count: number;
  current_page: number;
  total_pages: number;
}

export interface DeletedKeyResponse extends KeyResponse {
  deleted_at: string;
  deleted_by: string;
}

export interface DeletedKeysResponse {
  keys: DeletedKeyResponse[];
  total_count: number;
  current_page: number;
  total_pages: number;
}

export interface KeyListCallOptions {
  organizationID?: string | null;
  teamID?: string | null;
  selectedKeyAlias?: string | null;
  userID?: string | null;
  keyHash?: string | null;
  sortBy?: string | null;
  sortOrder?: string | null;
  expand?: string | null;
  status?: string | null;
}

const keyListCall = async (
  accessToken: string,
  page: number,
  pageSize: number,
  options: KeyListCallOptions = {},
) => {
  /**
   * Get all available keys on proxy
   */
  try {
    const baseUrl = getProxyBaseUrl();
    
    const params = new URLSearchParams(
      Object.entries({
        team_id: options.teamID,
        organization_id: options.organizationID,
        key_alias: options.selectedKeyAlias,
        key_hash: options.keyHash,
        user_id: options.userID,
        page,
        size: pageSize,
        sort_by: options.sortBy,
        sort_order: options.sortOrder,
        expand: options.expand,
        status: options.status,
        return_full_object: "true",
        include_team_keys: "true",
        include_created_by_keys: "true",
      })
        .filter(([, value]) => value !== undefined && value !== null)
        .map(([key, value]) => [key, String(value)]),
    );

    const url = `${baseUrl ? `${baseUrl}/key/list` : "/key/list"}?${params}`;

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
    console.log("/key/list API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to list keys:", error);
    throw error;
  }
};

export const useKeys = (
  page: number,
  pageSize: number,
  options: KeyListCallOptions = {},
): UseQueryResult<KeysResponse> => {
  const { accessToken } = useAuthorized();

  return useQuery<KeysResponse>({
    queryKey: keyKeys.list({ page, limit: pageSize, ...options }),
    queryFn: async () => await keyListCall(accessToken!, page, pageSize, options),
    enabled: Boolean(accessToken),
    staleTime: 30000, // 30 seconds
    placeholderData: keepPreviousData,
  });
};

export const deletedKeyKeys = createQueryKeys("deletedKeys");
export const useDeletedKeys = (
  page: number,
  pageSize: number,
  options: KeyListCallOptions = {},
): UseQueryResult<KeysResponse> => {
  const { accessToken } = useAuthorized();

  return useQuery<KeysResponse>({
    queryKey: deletedKeyKeys.list({ page, limit: pageSize, ...options }),
    queryFn: async () => await keyListCall(accessToken!, page, pageSize, { ...options, status: "deleted" }),
    enabled: Boolean(accessToken),
    staleTime: 30000, // 30 seconds
    placeholderData: keepPreviousData,
  });
};