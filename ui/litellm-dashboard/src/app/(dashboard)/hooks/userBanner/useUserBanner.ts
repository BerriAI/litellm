import { getUserBanner, UserBanner } from "@/components/networking";
import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

export const userBannerKeys = createQueryKeys("userBanner");

export const useUserBanner = (accessToken: string | null) => {
  const queryOptions: UseQueryOptions<UserBanner> = {
    queryKey: userBannerKeys.list({}),
    queryFn: async () => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await getUserBanner(accessToken);
    },
    enabled: Boolean(accessToken),
    staleTime: 60 * 1000,
    gcTime: 5 * 60 * 1000,
  };
  return useQuery<UserBanner>(queryOptions);
};
