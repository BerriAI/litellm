import { keepPreviousData, useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { keyListCall } from "@/components/networking";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const keyKeys = createQueryKeys("keys");

export interface KeysResponse {
  keys: KeyResponse[];
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
