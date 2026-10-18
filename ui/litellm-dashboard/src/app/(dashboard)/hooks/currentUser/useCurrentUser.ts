import { skipToken, useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { fetchClient } from "@/lib/http/api";
import { getAuthHeaderName } from "@/lib/http/runtime";
import type { components } from "@/lib/http/schema";
import { createQueryKeys } from "../common/queryKeysFactory";

export type CurrentUser = components["schemas"]["CurrentUserResponse"];

const currentUserKeys = createQueryKeys("currentUser");

const getCurrentUser = async (accessToken: string, authHeaderName: string): Promise<CurrentUser> => {
  const { data } = await fetchClient.GET("/management/v1/users/me", {
    headers: { [authHeaderName]: `Bearer ${accessToken}` },
  });
  if (!data) throw new Error("Current user response is empty");
  return data;
};

export const useCurrentUser = (accessToken: string | null, authHeaderName?: string, canFetch = true) => {
  const options: UseQueryOptions<CurrentUser> = {
    queryKey: currentUserKeys.detail(accessToken ?? ""),
    queryFn:
      accessToken && canFetch ? () => getCurrentUser(accessToken, authHeaderName ?? getAuthHeaderName()) : skipToken,
    staleTime: Infinity,
    retry: 1,
  };
  return useQuery(options);
};
