import moment from "moment";
import { useCallback, useEffect, useState, useRef, useMemo } from "react";
import { uiSpendLogsCall } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { useQuery } from "@tanstack/react-query";
import { fetchAllTeams } from "../../components/key_team_helpers/filter_helpers";
import { debounce } from "lodash";
import { defaultPageSize } from "../constants";
import { PaginatedResponse } from ".";
import type { LogsSortField } from "./columns";

const FILTER_KEYS = {
  TEAM_ID: "Team ID",
  KEY_HASH: "Key Hash",
  REQUEST_ID: "Request ID",
  MODEL: "Model",
  USER_ID: "User ID",
  END_USER: "End User",
  STATUS: "Status",
  KEY_ALIAS: "Key Alias",
  ERROR_CODE: "Error Code",
  ERROR_MESSAGE: "Error Message",
} as const;

export type FilterKey = keyof typeof FILTER_KEYS;
export type LogFilterState = Record<(typeof FILTER_KEYS)[FilterKey], string>;

export function useLogFilterLogic({
  logs,
  accessToken,
  startTime, // Receive from SpendLogsTable
  endTime, // Receive from SpendLogsTable
  pageSize = defaultPageSize,
  isCustomDate,
  setCurrentPage,
  userID,
  userRole,
  sortBy = "startTime",
  sortOrder = "desc",
  currentPage = 1,
}: {
  logs: PaginatedResponse;
  accessToken: string | null;
  startTime: string;
  endTime: string;
  pageSize?: number;
  isCustomDate: boolean;
  setCurrentPage: (page: number) => void;
  userID: string | null;
  userRole: string | null;
  sortBy?: LogsSortField;
  sortOrder?: "asc" | "desc";
  currentPage?: number;
}) {
  const defaultFilters = useMemo<LogFilterState>(
    () => ({
      [FILTER_KEYS.TEAM_ID]: "",
      [FILTER_KEYS.KEY_HASH]: "",
      [FILTER_KEYS.REQUEST_ID]: "",
      [FILTER_KEYS.MODEL]: "",
      [FILTER_KEYS.USER_ID]: "",
      [FILTER_KEYS.END_USER]: "",
      [FILTER_KEYS.STATUS]: "",
      [FILTER_KEYS.KEY_ALIAS]: "",
      [FILTER_KEYS.ERROR_CODE]: "",
      [FILTER_KEYS.ERROR_MESSAGE]: "",
    }),
    [],
  );

  const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
  const [backendFilteredLogs, setBackendFilteredLogs] = useState<PaginatedResponse | null>(null);
  const lastSearchTimestamp = useRef(0);

  // Refs that always hold the latest filters and hasBackendFilters values.
  // The sort/page/time effect below intentionally omits these from its dep array
  // to avoid double-fetches when a filter changes; reading from refs instead of
  // the closure prevents stale-closure bugs (e.g. the effect using a snapshot of
  // filters taken before the user selected Key Alias).
  const filtersRef = useRef(filters);
  const hasBackendFiltersRef = useRef(false);
  const performSearch = useCallback(
    async (filters: LogFilterState, page = 1) => {
      if (!accessToken) return;

      console.log("Filters being sent to API:", filters);
      const currentTimestamp = Date.now();
      lastSearchTimestamp.current = currentTimestamp;

      const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = isCustomDate
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      try {
        const response = await uiSpendLogsCall({
          accessToken,
          start_date: formattedStartTime,
          end_date: formattedEndTime,
          page,
          page_size: pageSize,
          params: {
            api_key: filters[FILTER_KEYS.KEY_HASH] || undefined,
            team_id: filters[FILTER_KEYS.TEAM_ID] || undefined,
            request_id: filters[FILTER_KEYS.REQUEST_ID] || undefined,
            user_id: filters[FILTER_KEYS.USER_ID] || undefined,
            end_user: filters[FILTER_KEYS.END_USER] || undefined,
            status_filter: filters[FILTER_KEYS.STATUS] || undefined,
            model_id: filters[FILTER_KEYS.MODEL] || undefined,
            key_alias: filters[FILTER_KEYS.KEY_ALIAS] || undefined,
            error_code: filters[FILTER_KEYS.ERROR_CODE] || undefined,
            error_message: filters[FILTER_KEYS.ERROR_MESSAGE] || undefined,
            sort_by: sortBy,
            sort_order: sortOrder,
          },
        });

        if (currentTimestamp === lastSearchTimestamp.current) {
          setBackendFilteredLogs({
            ...response,
            data: response.data ?? [],
          });
        }
      } catch (error) {
        console.error("Error searching users:", error);
        setBackendFilteredLogs({
          data: [],
          total: 0,
          page: 1,
          page_size: pageSize,
          total_pages: 0,
        });
      }
    },
    [accessToken, startTime, endTime, isCustomDate, pageSize, sortBy, sortOrder],
  );

  const debouncedSearch = useMemo(
    () => debounce((filters: LogFilterState, page: number) => performSearch(filters, page), 300),
    [performSearch],
  );

  useEffect(() => {
    return () => debouncedSearch.cancel();
  }, [debouncedSearch]);

  // Determine when backend filters are active (server-side filtering)
  const hasBackendFilters = useMemo(
    () =>
      !!(
        filters[FILTER_KEYS.KEY_ALIAS] ||
        filters[FILTER_KEYS.KEY_HASH] ||
        filters[FILTER_KEYS.REQUEST_ID] ||
        filters[FILTER_KEYS.USER_ID] ||
        filters[FILTER_KEYS.END_USER] ||
        filters[FILTER_KEYS.ERROR_CODE] ||
        filters[FILTER_KEYS.ERROR_MESSAGE] ||
        filters[FILTER_KEYS.MODEL]
      ),
    [filters],
  );

  // Keep refs in sync on every render so the sort/page/time effect always reads
  // the latest values without those values being in its dep array.
  useEffect(() => {
    filtersRef.current = filters;
    hasBackendFiltersRef.current = hasBackendFilters;
  }, [filters, hasBackendFilters]);

  // Refetch when sort, page, or time range changes (backend filters use their own fetch, not the main query)
  useEffect(() => {
    if (hasBackendFiltersRef.current && accessToken) {
      // Cancel any pending debounced search to prevent it from overwriting this page's results
      debouncedSearch.cancel();
      performSearch(filtersRef.current, currentPage);
    }
    // filters / hasBackendFilters are read via refs — avoids stale-closure bugs
    // when sort/page/time changes after a filter (e.g. Key Alias) was set.
    // debouncedSearch / performSearch: filter changes go through handleFilterChange
    // → debouncedSearch; adding them here would cause double-fetches on filter apply.
    // accessToken: stable across sort/page/time changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortBy, sortOrder, currentPage, startTime, endTime, isCustomDate]);

  // Compute client-side filtered logs directly from incoming logs and filters
  const clientDerivedFilteredLogs: PaginatedResponse = useMemo(() => {
    if (!logs || !logs.data) {
      return {
        data: [],
        total: 0,
        page: 1,
        page_size: pageSize,
        total_pages: 0,
      };
    }

    // If backend filters are on, don't perform client-side filtering here
    if (hasBackendFilters) {
      return logs;
    }

    let filteredData = [...logs.data];

    if (filters[FILTER_KEYS.TEAM_ID]) {
      filteredData = filteredData.filter((log) => log.team_id === filters[FILTER_KEYS.TEAM_ID]);
    }

    if (filters[FILTER_KEYS.STATUS]) {
      filteredData = filteredData.filter((log) => {
        if (filters[FILTER_KEYS.STATUS] === "success") {
          return !log.status || log.status === "success";
        }
        return log.status === filters[FILTER_KEYS.STATUS];
      });
    }

    if (filters[FILTER_KEYS.MODEL]) {
      filteredData = filteredData.filter((log) => log.model_id === filters[FILTER_KEYS.MODEL]);
    }

    if (filters[FILTER_KEYS.KEY_HASH]) {
      filteredData = filteredData.filter((log) => log.api_key === filters[FILTER_KEYS.KEY_HASH]);
    }

    if (filters[FILTER_KEYS.END_USER]) {
      filteredData = filteredData.filter((log) => log.end_user === filters[FILTER_KEYS.END_USER]);
    }

    if (filters[FILTER_KEYS.ERROR_CODE]) {
      filteredData = filteredData.filter((log) => {
        const metadata = log.metadata || {};
        const errorInfo = metadata.error_information;
        return errorInfo && errorInfo.error_code === filters[FILTER_KEYS.ERROR_CODE];
      });
    }

    return {
      data: filteredData,
      total: logs.total,
      page: logs.page,
      page_size: logs.page_size,
      total_pages: logs.total_pages,
    };
  }, [logs, filters, hasBackendFilters]);

  // Choose which filtered logs to expose: backend result when active, otherwise client-derived
  const filteredLogs: PaginatedResponse = useMemo(() => {
    if (hasBackendFilters) {
      // When backend filters are active, only show backend results.
      // If search hasn't completed yet (null), show empty state rather than
      // falling back to unfiltered logs — that caused filtered views to
      // display mismatched data when the filter matched zero rows.
      if (backendFilteredLogs !== null) {
        return backendFilteredLogs;
      }
      return {
        data: [],
        total: 0,
        page: 1,
        page_size: pageSize,
        total_pages: 0,
      };
    }
    return clientDerivedFilteredLogs;
  }, [hasBackendFilters, backendFilteredLogs, clientDerivedFilteredLogs]);

  // Fetch all teams and users for potential filter dropdowns (optional, can be adapted)
  const { data: allTeams } = useQuery<Team[], Error>({
    queryKey: ["allTeamsForLogFilters", accessToken],
    queryFn: async () => {
      if (!accessToken) return [];
      // Use fetchAllTeams helper function for consistency and abstraction
      // Assuming fetchAllTeams returns Team[] directly
      const teamsData = await fetchAllTeams(accessToken);
      return teamsData || []; // Ensure it returns an array
    },
    enabled: !!accessToken,
  });

  // Update filters state
  const handleFilterChange = (newFilters: Partial<LogFilterState>) => {
    setFilters((prev) => {
      const updatedFilters = { ...prev, ...newFilters };

      // Ensure all keys in LogFilterState are present, defaulting to '' if not in newFilters
      for (const key of Object.keys(defaultFilters) as Array<keyof LogFilterState>) {
        if (!(key in updatedFilters)) {
          updatedFilters[key] = defaultFilters[key];
        }
      }

      // Only call debouncedSearch if filters have actually changed
      if (JSON.stringify(updatedFilters) !== JSON.stringify(prev)) {
        setCurrentPage(1);
        setBackendFilteredLogs(null);
        debouncedSearch(updatedFilters, 1);
      }

      return updatedFilters as LogFilterState;
    });
  };

  const handleFilterReset = () => {
    // Reset filters state
    setFilters(defaultFilters);

    // Clear backend filtered logs to ensure fresh render
    setBackendFilteredLogs(null);

    // Cancel any in-flight debounced search
    debouncedSearch.cancel();

    // Reset to first page so the unfiltered view starts at page 1
    setCurrentPage(1);
  };

  // Expose a filter-aware refetch so callers (e.g. the manual Fetch button) can
  // refresh results while keeping all active backend filters intact.  The plain
  // `logs.refetch()` in the parent only re-runs the main TanStack Query, which
  // does not carry key_alias or other backend-only filter params.
  const refetchWithFilters = useCallback(
    (page = currentPage) => {
      if (hasBackendFilters && accessToken) {
        debouncedSearch.cancel();
        performSearch(filters, page);
      }
    },
    [hasBackendFilters, accessToken, filters, currentPage, performSearch, debouncedSearch],
  );

  return {
    filters,
    filteredLogs,
    hasBackendFilters,
    allTeams,
    handleFilterChange,
    handleFilterReset,
    refetchWithFilters,
  };
}
