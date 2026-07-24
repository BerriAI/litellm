import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

type EndUserAliasesPage = components["schemas"]["CustomerAliasesResponse"];

export interface EndUserAliasesWindow {
  start_date: string;
  end_date: string;
}

export const useInfiniteEndUserAliases = (window: EndUserAliasesWindow, size: number = 50, search?: string) => {
  const { accessToken } = useAuthorized();
  const query = { ...window, size, ...(search !== undefined && search !== "" ? { search } : {}) };
  const options = {
    pageParamName: "page",
    initialPageParam: 1,
    getNextPageParam: (lastPage: EndUserAliasesPage) => (lastPage.has_more ? lastPage.current_page + 1 : undefined),
    enabled: Boolean(accessToken),
  };
  return $api.useInfiniteQuery("get", "/customer/aliases", { params: { query } }, options);
};
