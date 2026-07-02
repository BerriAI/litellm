import { $api, authHeader } from "@/lib/http/api";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const DEFAULT_PAGE_SIZE = 50;

export const useInfiniteUsers = (pageSize: number = DEFAULT_PAGE_SIZE, searchEmail?: string) => {
  const { accessToken, userRole } = useAuthorized();
  return $api.useInfiniteQuery(
    "get",
    "/user/list",
    {
      params: {
        query: {
          page_size: pageSize,
          ...(searchEmail ? { user_email: searchEmail } : {}),
        },
      },
      headers: authHeader(accessToken!),
    },
    {
      pageParamName: "page",
      initialPageParam: 1,
      getNextPageParam: (lastPage) => (lastPage.page < lastPage.total_pages ? lastPage.page + 1 : undefined),
      enabled: Boolean(accessToken) && all_admin_roles.includes(userRole!),
    },
  );
};
