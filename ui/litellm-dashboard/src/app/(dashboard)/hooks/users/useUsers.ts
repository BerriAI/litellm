import { userListCall, UserListResponse } from "@/components/networking";
import { useInfiniteQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const infiniteUsersKeys = createQueryKeys("infiniteUsers");

const DEFAULT_PAGE_SIZE = 50;

export const useInfiniteUsers = (
  pageSize: number = DEFAULT_PAGE_SIZE,
  searchEmail?: string,
) => {
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
        null,                       // userIDs
        pageParam as number,        // page
        pageSize,                   // page_size
        searchEmail || null,        // userEmail
      );
    },
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      if (lastPage.page < lastPage.total_pages) {
        return lastPage.page + 1;
      }
      return undefined;
    },
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole!),
  });
};
