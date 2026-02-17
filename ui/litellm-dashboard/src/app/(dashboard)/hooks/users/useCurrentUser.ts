import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { UserInfo, userInfoCall } from "@/components/networking";
import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const userKeys = createQueryKeys("users");

export const useCurrentUser = (): UseQueryResult<UserInfo> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<UserInfo>({
    queryKey: userKeys.detail(userId!),
    queryFn: async () => {
      const data = await userInfoCall(accessToken!, userId!, userRole!, false, null, null);
      console.log(`userInfo: ${JSON.stringify(data)}`);
      return data.user_info;
    },
    enabled: Boolean(accessToken && userId && userRole),
  });
};
