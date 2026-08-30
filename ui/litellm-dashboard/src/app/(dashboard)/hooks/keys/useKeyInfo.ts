import { useQuery, UseQueryResult } from "@tanstack/react-query";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { keyInfoV1Call } from "@/components/networking";

import { keyKeys } from "./useKeys";

export function useKeyInfo(keyId: string | null, options?: { enabled?: boolean }): UseQueryResult<KeyResponse> {
  const { accessToken } = useAuthorized();

  return useQuery<KeyResponse>({
    queryKey: [...keyKeys.detail(keyId ?? ""), accessToken],
    queryFn: async () => {
      if (!accessToken || !keyId) throw new Error("Missing access token or key id");
      const keyData = await keyInfoV1Call(accessToken, keyId);
      return {
        ...keyData["info"],
        token: keyId,
        api_key: keyId,
      };
    },
    enabled: Boolean(accessToken && keyId) && (options?.enabled ?? true),
  });
}
