import { userGetInfoV2, userListCall } from "@/components/networking";
import type { UserInfoV2Response, UserListResponse } from "@/components/networking";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { canListUsers } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ApiError } from "@/lib/http/client";
import { useCurrentUser } from "../currentUser/useCurrentUser";

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

export type UserIdentityLookup = Readonly<
  Record<string, Readonly<Pick<UserInfoV2Response, "user_email" | "user_alias">>>
>;

export const userIdentityLabel = (id: string | null | undefined, identities: UserIdentityLookup): string => {
  if (!id) return "";
  return identities[id]?.user_email?.trim() || identities[id]?.user_alias?.trim() || id;
};

export const useUserIdentityLookup = (
  userIds: readonly (string | null | undefined)[],
  scope: { accessToken: string | null; userRole?: string; isViewOnly?: boolean },
): UserIdentityLookup => {
  const { accessToken, userRole, isViewOnly = false } = scope;
  const currentUser = useCurrentUser(accessToken);
  const me = currentUser.isPlaceholderData ? undefined : currentUser.data;
  const effectiveRole = userRole ?? me?.user_role ?? "";
  const canLookupOthers = canListUsers(effectiveRole);
  const ids = Array.from(new Set(userIds.filter((id): id is string => typeof id === "string" && id !== ""))).sort();
  const lookupIds = canLookupOthers ? ids : ids.filter((id) => id === me?.user_id);
  const { data, isPlaceholderData } = useQuery<UserIdentityLookup>({
    queryKey: [
      "userIdentityLookup",
      { accessToken, effectiveRole, isViewOnly, currentUserRole: me?.user_role, currentUserId: me?.user_id, ids },
    ],
    queryFn: async () => {
      if (!accessToken) return {};
      const batches = Array.from({ length: Math.ceil(lookupIds.length / USER_LIST_MAX_PAGE_SIZE) }, (_, index) =>
        lookupIds.slice(index * USER_LIST_MAX_PAGE_SIZE, (index + 1) * USER_LIST_MAX_PAGE_SIZE),
      );
      const users = await Promise.all(
        batches.map(async (batch) => {
          try {
            if (batch.length === 1) {
              const user = await userGetInfoV2(accessToken, batch[0]);
              return user.user_id === batch[0] ? [user] : [];
            }
            const response = await userListCall(accessToken, batch, 1, batch.length);
            return response.users.filter((user) => batch.includes(user.user_id));
          } catch (error) {
            if (error instanceof ApiError && (error.status === 403 || error.status === 404)) return [];
            throw error;
          }
        }),
      );
      return Object.fromEntries(
        users
          .flat()
          .map((user) => [user.user_id, { user_email: user.user_email ?? null, user_alias: user.user_alias ?? null }]),
      );
    },
    enabled: Boolean(accessToken) && lookupIds.length > 0,
  });

  if (!accessToken) return {};
  const identities = isPlaceholderData ? {} : data ?? {};
  if (!me || !ids.includes(me.user_id)) return identities;
  const self = identities[me.user_id] ?? { user_email: null, user_alias: null };
  return {
    ...identities,
    [me.user_id]: {
      ...self,
      user_email: me.user_email?.trim() || self.user_email,
    },
  };
};

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
  return useQuery<UserInfoV2Response | null>({
    queryKey: userLookupKeys.detail(userId ?? ""),
    queryFn: async () => {
      try {
        const user = await userGetInfoV2(accessToken!, userId!);
        return user.user_id === userId ? user : null;
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    enabled: Boolean(accessToken) && Boolean(userId) && canListUsers(userRole),
  });
};
