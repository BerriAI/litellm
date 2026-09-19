import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { UserInfoV2Response, userGetInfoV2 } from "@/components/networking";
import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const userKeys = createQueryKeys("users");

/** Fetch a specific user's info from `/v2/user/info?user_id=`; disabled when userId is null. */
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
