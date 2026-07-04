import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { uiSpendLogsCall } from "@/components/networking";
import type { AdminObservabilityFilters } from "./types";

interface UseAdminObservabilityOptions {
  accessToken: string | null;
  startDate: string;
  endDate: string;
  page: number;
  pageSize: number;
  filters: AdminObservabilityFilters;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

const buildQueryKey = (options: Omit<UseAdminObservabilityOptions, "accessToken">) => [
  "admin-observability",
  options.startDate,
  options.endDate,
  options.page,
  options.pageSize,
  options.filters,
  options.sortBy,
  options.sortOrder,
];

export function useAdminObservability({
  accessToken,
  startDate,
  endDate,
  page,
  pageSize,
  filters,
  sortBy = "startTime",
  sortOrder = "desc",
}: UseAdminObservabilityOptions) {
  const params = useMemo(
    () => ({
      ...filters,
      sort_by: sortBy,
      sort_order: sortOrder,
    }),
    [filters, sortBy, sortOrder],
  );

  return useQuery({
    queryKey: buildQueryKey({ startDate, endDate, page, pageSize, filters, sortBy, sortOrder }),
    queryFn: async () => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return uiSpendLogsCall({
        accessToken,
        start_date: startDate,
        end_date: endDate,
        page,
        page_size: pageSize,
        params,
      });
    },
    enabled: !!accessToken,
    placeholderData: (previousData) => previousData,
  });
}
