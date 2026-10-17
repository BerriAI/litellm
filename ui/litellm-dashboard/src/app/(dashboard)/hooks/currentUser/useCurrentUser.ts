import { apiClient } from "@/components/networking";
import { skipToken, useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

export type CurrentUser = {
  user_id: string;
  user_role: string | null;
  user_email: string | null;
};

const currentUserKeys = createQueryKeys("currentUser");

export const getCurrentUser = (accessToken: string): Promise<CurrentUser> =>
  apiClient.get<CurrentUser>("/management/v1/users/me", { accessToken });

export const useCurrentUser = (accessToken: string | null) => {
  const options: UseQueryOptions<CurrentUser> = {
    queryKey: currentUserKeys.detail(accessToken ?? ""),
    queryFn: accessToken ? () => getCurrentUser(accessToken) : skipToken,
    staleTime: Infinity,
    retry: 1,
  };
  return useQuery(options);
};
