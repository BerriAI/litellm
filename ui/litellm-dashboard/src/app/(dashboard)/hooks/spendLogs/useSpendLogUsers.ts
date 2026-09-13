import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";

import { nextPageFromLinks, type SpendLogsWindow } from "./useSpendLogEndUsers";

export const useInfiniteSpendLogUsers = (window: SpendLogsWindow, pageSize: number = 50, q?: string) => {
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
  return $api.useInfiniteQuery("get", "/management/v1/spend_logs/users", { params: { query } }, options);
};
