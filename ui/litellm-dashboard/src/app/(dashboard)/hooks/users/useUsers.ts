import { userListCall, UserInfo, UserListResponse } from "@/components/networking";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { canListUsers } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const infiniteUsersKeys = createQueryKeys("infiniteUsers");
const userLookupKeys = createQueryKeys("userLookup");

const DEFAULT_PAGE_SIZE = 50;

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

const userDisplayNameKeys = createQueryKeys("userDisplayNames");

export const useUserDisplayNames = (userIds: readonly string[]) => {
  const { accessToken, userRole } = useAuthorized();
  const distinctIds = Array.from(new Set(userIds.filter((id) => id !== ""))).sort();
  return useQuery<Record<string, string>>({
    queryKey: userDisplayNameKeys.list({ filters: { ids: JSON.stringify(distinctIds) } }),
    queryFn: async () => {
      const chunks: string[][] = [];
      for (let i = 0; i < distinctIds.length; i += USER_LIST_MAX_PAGE_SIZE) {
        chunks.push(distinctIds.slice(i, i + USER_LIST_MAX_PAGE_SIZE));
      }
      const responses = await Promise.all(
        chunks.map((ids) =>
          // a single user_ids value is a substring search on /user/list
          userListCall(accessToken!, ids.length === 1 ? [ids[0], ids[0]] : ids, 1, USER_LIST_MAX_PAGE_SIZE),
        ),
      );
      return Object.fromEntries(
        responses.flatMap((response, chunkIndex) => {
          const requested = new Set(chunks[chunkIndex]);
          return response.users.flatMap((user) => {
            if (!requested.has(user.user_id)) return [];
            const name = user.user_alias || user.user_email;
            return name ? [[user.user_id, name]] : [];
          });
        }),
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
      const response = await userListCall(accessToken!, [userId!], 1, 1);
      return response.users.find((user) => user.user_id === userId) ?? null;
    },
    enabled: Boolean(accessToken) && Boolean(userId) && canListUsers(userRole),
  });
};
