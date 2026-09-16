import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { UserInfoV2Response, userGetInfoV2 } from "@/components/networking";
import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const userKeys = createQueryKeys("users");

/**
 * Fetch a SPECIFIC user's info from /v2/user/info?user_id=<userId>.
 *
 * Companion to useCurrentUser (which self-looks-up the caller): pass the user
 * you actually want to display. Disabled when userId is null so the caller can
 * fall back to a global/unfiltered view without firing a request. Admins may
 * query any user; the backend authorizes the lookup.
 */
export const useUserInfo = (userId: string | null): UseQueryResult<UserInfoV2Response> => {
  const { accessToken } = useAuthorized();
  return useQuery<UserInfoV2Response>({
    queryKey: userKeys.detail(userId ?? ""),
    queryFn: async () => {
      return await userGetInfoV2(accessToken!, userId!);
    },
    enabled: Boolean(accessToken && userId),
  });
};
