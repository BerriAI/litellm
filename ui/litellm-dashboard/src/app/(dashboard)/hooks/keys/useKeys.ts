import { keepPreviousData, useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { keyListCall } from "@/components/networking";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export const keyKeys = createQueryKeys("keys");

export interface KeysResponse {
  keys: KeyResponse[];
  total_count: number;
  current_page: number;
  total_pages: number;
}

export interface DeletedKeyResponse {
  token: string;
  token_id: string;
  key_name: string;
  key_alias: string;
}

export interface DeletedKeysResponse {
  keys: DeletedKeyResponse[];
  total_count: number;
  current_page: number;
  total_pages: number;
}

export const useKeys = (page: number, pageSize: number): UseQueryResult<KeysResponse> => {
  const { accessToken } = useAuthorized();

  return useQuery<KeysResponse>({
    queryKey: keyKeys.list({ page, limit: pageSize }),
    queryFn: async () =>
      await keyListCall(
        accessToken!,
        null, // organizationID
        null, // teamID
        null, // selectedKeyAlias
        null, // userID
        null, // keyHash
        page,
        pageSize,
      ),
    enabled: Boolean(accessToken),
    staleTime: 30000, // 30 seconds
    placeholderData: keepPreviousData,
  });
};

export const deletedKeyKeys = createQueryKeys("deletedKeys");
export const useDeletedKeys = (page: number, pageSize: number): UseQueryResult<KeysResponse> => {
  const { accessToken } = useAuthorized();

  return useQuery<KeysResponse>({
    queryKey: deletedKeyKeys.list({ page, limit: pageSize }),
    queryFn: async () =>
      await keyListCall(accessToken!, null, null, null, null, null, page, pageSize, null, null, null, "deleted"),
    enabled: Boolean(accessToken),
    staleTime: 30000, // 30 seconds
    placeholderData: keepPreviousData,
  });
};