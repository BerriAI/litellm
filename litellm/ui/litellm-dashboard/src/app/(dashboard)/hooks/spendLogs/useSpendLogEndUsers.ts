import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

type EndUsersPage = components["schemas"]["FacetListResponse"];

export interface SpendLogsWindow {
  start_date: string;
  end_date: string;
}

/** Reads the server's `links.next` instead of computing the next page, so the
 *  endpoint can move to cursor pagination without touching this hook. */
export const nextPageFromLinks = (lastPage: EndUsersPage): number | undefined => {
  const next = lastPage.links.next;
  if (!next) return undefined;
  const page = new URLSearchParams(next.slice(next.indexOf("?") + 1)).get("page");
  return page === null ? undefined : Number(page);
};

export const useInfiniteSpendLogEndUsers = (window: SpendLogsWindow, pageSize: number = 50, q?: string) => {
  const { accessToken } = useAuthorized();
  const query = {
    "filter[startTime][gte]": window.start_date,
    "filter[startTime][lte]": window.end_date,
    page_size: pageSize,
    ...(q !== undefined && q !== "" ? { q } : {}),
  };
  const options = {
    pageParamName: "page",
    initialPageParam: 1,
    getNextPageParam: nextPageFromLinks,
    enabled: Boolean(accessToken),
  };
  return $api.useInfiniteQuery("get", "/management/v1/spend_logs/end_users", { params: { query } }, options);
};
