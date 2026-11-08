import { useQuery, UseQueryResult } from "@tanstack/react-query";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { keyInfoV1Call, userGetInfoV2 } from "@/components/networking";

import { keyKeys } from "./useKeys";

const fetchOwner = async (accessToken: string, userId: string): Promise<KeyResponse["user"] | undefined> => {
  try {
    const owner = await userGetInfoV2(accessToken, userId);
    return {
      user_id: owner.user_id,
      user_email: owner.user_email,
      user_alias: owner.user_alias,
      max_budget: owner.max_budget,
      budget_duration: owner.budget_duration,
    };
  } catch {
    return undefined;
  }
};

export function useKeyInfo(keyId: string | null, options?: { enabled?: boolean }): UseQueryResult<KeyResponse> {
  const { accessToken } = useAuthorized();

  return useQuery<KeyResponse>({
    queryKey: [...keyKeys.detail(keyId ?? ""), accessToken],
    queryFn: async () => {
      if (!accessToken || !keyId) throw new Error("Missing access token or key id");
      const keyData = await keyInfoV1Call(accessToken, keyId);
      const info = keyData["info"];
      const owner =
        typeof info.user_id === "string" && info.user_id !== ""
          ? await fetchOwner(accessToken, info.user_id)
          : undefined;
      return {
        ...info,
        token: keyId,
        api_key: keyId,
        ...(owner ? { user: owner } : {}),
      };
    },
    enabled: Boolean(accessToken && keyId) && (options?.enabled ?? true),
  });
}
