import { userGetInfoV2, userListCall } from "@/components/networking";
import type { UserInfo, UserInfoV2Response, UserListResponse } from "@/components/networking";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { canListUsers } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ApiError } from "@/lib/http/client";

const infiniteUsersKeys = createQueryKeys("infiniteUsers");
const userLookupKeys = createQueryKeys("userLookup");

const DEFAULT_PAGE_SIZE = 50;

const mapUserInfoV2ToUserInfo = (user: UserInfoV2Response): UserInfo => {
  const userInfo: UserInfo = {
    user_id: user.user_id,
    user_email: user.user_email,
    user_alias: user.user_alias,
    user_role: user.user_role,
    spend: user.spend,
    max_budget: user.max_budget,
    models: user.models,
    budget_duration: user.budget_duration,
    budget_reset_at: user.budget_reset_at,
    metadata: user.metadata,
    created_at: user.created_at,
    updated_at: user.updated_at,
    sso_user_id: user.sso_user_id,
    teams: user.teams,
    tpm_limit: user.tpm_limit,
    rpm_limit: user.rpm_limit,
    allowed_cache_controls: [],
    key_count: 0,
    model_max_budget: null,
    model_spend: null,
    policies: [],
  };
  return userInfo;
};

export const useInfiniteUsers = (pageSize: number = DEFAULT_PAGE_SIZE, searchEmail?: string) => {
  const { accessToken, userRole } = useAuthorized();
  return useInfiniteQuery<UserListResponse>({
    queryKey: infiniteUsersKeys.list({
      filters: {
        pageSize,
        ...(searchEmail && { searchEmail }),
      },
    }),
    queryFn: async ({ pageParam }) => {
      return await userListCall(
        accessToken!,
        null, // userIDs
        pageParam as number, // page
        pageSize, // page_size
        searchEmail || null, // userEmail
      );
    },
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      if (lastPage.page < lastPage.total_pages) {
        return lastPage.page + 1;
      }
      return undefined;
    },
    enabled: Boolean(accessToken) && canListUsers(userRole),
  });
};

const USER_LIST_MAX_PAGE_SIZE = 100;

export const useUserEmailLookup = (userIds: readonly string[]) => {
  const { accessToken, userRole } = useAuthorized();
  const distinctIds = Array.from(new Set(userIds.filter((id) => id !== ""))).sort();
  return useQuery<Record<string, string>>({
    queryKey: userLookupKeys.list({ filters: { ids: JSON.stringify(distinctIds) } }),
    queryFn: async () => {
      const ids = distinctIds.slice(0, USER_LIST_MAX_PAGE_SIZE);
      const response = await userListCall(accessToken!, ids, 1, ids.length);
      return Object.fromEntries(
        response.users.flatMap((user) => (user.user_email ? [[user.user_id, user.user_email]] : [])),
      );
    },
    enabled: Boolean(accessToken) && distinctIds.length > 0 && canListUsers(userRole),
  });
};

export const useUserLookup = (userId: string | null) => {
  const { accessToken, userRole } = useAuthorized();
  return useQuery<UserInfo | null>({
    queryKey: userLookupKeys.detail(userId ?? ""),
    queryFn: async () => {
      try {
        return mapUserInfoV2ToUserInfo(await userGetInfoV2(accessToken!, userId!));
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          const response = await userListCall(accessToken!, [userId!], 1, USER_LIST_MAX_PAGE_SIZE);
          return response.users.find((user) => user.user_id === userId) ?? null;
        }
        throw error;
      }
    },
    enabled: Boolean(accessToken) && Boolean(userId) && canListUsers(userRole),
  });
};
