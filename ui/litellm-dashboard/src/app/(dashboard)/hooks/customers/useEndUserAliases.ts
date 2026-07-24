import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { all_admin_roles } from "@/utils/roles";

type EndUserAliasesPage = components["schemas"]["CustomerAliasesResponse"];

export const useInfiniteEndUserAliases = (size: number = 50, search?: string) => {
  const { accessToken, userRole } = useAuthorized();
  const query = { size, ...(search !== undefined && search !== "" ? { search } : {}) };
  const options = {
    pageParamName: "page",
    initialPageParam: 1,
    getNextPageParam: (lastPage: EndUserAliasesPage) => (lastPage.has_more ? lastPage.current_page + 1 : undefined),
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole ?? ""),
  };
  return $api.useInfiniteQuery("get", "/customer/aliases", { params: { query } }, options);
};
