import { useQueries } from "@tanstack/react-query";
import { $api } from "@/lib/http/api";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { components } from "@/lib/http/schema";

export type EndUser = components["schemas"]["CustomerResponse"];

export const useCustomers = (pageSize: number = 20, search?: string) => {
  const { accessToken, userRole } = useAuthorized();
  const options = {
    pageParamName: "page",
    initialPageParam: 1,
    getNextPageParam: (lastPage: EndUser[], _pages: EndUser[][], lastPageParam: unknown) =>
      lastPage.length === pageSize && typeof lastPageParam === "number" ? lastPageParam + 1 : undefined,
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole!),
  };
  return $api.useInfiniteQuery(
    "get",
    "/customer/list",
    { params: { query: { page_size: pageSize, ...(search ? { search } : {}) } } },
    options,
  );
};

export const useCustomerLookups = (customerIds: readonly string[]) => {
  const { accessToken, userRole } = useAuthorized();
  return useQueries({
    queries: Array.from(new Set(customerIds)).map((endUserId) =>
      $api.queryOptions(
        "get",
        "/customer/info",
        { params: { query: { end_user_id: endUserId } } },
        { enabled: Boolean(accessToken) && all_admin_roles.includes(userRole!) },
      ),
    ),
  });
};
