import moment from "moment";
import { useCallback, useEffect, useState, useRef, useMemo } from "react";
import { uiSpendLogsCall } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { fetchAllTeams } from "../../components/key_team_helpers/filter_helpers";
import { debounce } from "lodash";
import { defaultPageSize } from "../constants";
import { PaginatedResponse } from ".";
import type { LogsSortField } from "./columns";

export const FILTER_KEYS = {
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

export const defaultFilters: LogFilterState = {
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
};

export function useLogFilterLogic({
  accessToken,
  token,
  userRole,
  userID,
  filters,
  setFilters,
  filterByCurrentUser,
  activeTab,
  isLiveTail,
  startTime,
  endTime,
  pageSize = defaultPageSize,
  isCustomDate,
  setCurrentPage,
  sortBy = "startTime",
  sortOrder = "desc",
  currentPage = 1,
}: {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  filters: LogFilterState;
  setFilters: React.Dispatch<React.SetStateAction<LogFilterState>>;
  filterByCurrentUser: boolean | null;
  activeTab: string;
  isLiveTail: boolean;
  startTime: string;
  endTime: string;
  pageSize?: number;
  isCustomDate: boolean;
  setCurrentPage: (page: number) => void;
  sortBy?: LogsSortField;
  sortOrder?: "asc" | "desc";
  currentPage?: number;
}) {
  const [backendFilteredLogs, setBackendFilteredLogs] = useState<PaginatedResponse | null>(null);
  const lastSearchTimestamp = useRef(0);
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

  const logsQuery = useQuery<PaginatedResponse>({
    queryKey: [
      "logs",
      "table",
      currentPage,
      pageSize,
      startTime,
      endTime,
      filters[FILTER_KEYS.TEAM_ID],
      filters[FILTER_KEYS.KEY_HASH],
      filterByCurrentUser ? userID : null,
      filters[FILTER_KEYS.STATUS],
      filters[FILTER_KEYS.MODEL],
      sortBy,
      sortOrder,
    ],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        return {
          data: [],
          total: 0,
          page: 1,
          page_size: pageSize,
          total_pages: 0,
        };
      }

      const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = isCustomDate
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      const response = await uiSpendLogsCall({
        accessToken,
        start_date: formattedStartTime,
        end_date: formattedEndTime,
        page: currentPage,
        page_size: pageSize,
        params: {
          api_key: filters[FILTER_KEYS.KEY_HASH] || undefined,
          team_id: filters[FILTER_KEYS.TEAM_ID] || undefined,
          user_id: filterByCurrentUser ? userID ?? undefined : undefined,
          end_user: filters[FILTER_KEYS.END_USER] || undefined,
          status_filter: filters[FILTER_KEYS.STATUS] || undefined,
          model_id: filters[FILTER_KEYS.MODEL] || undefined,
          sort_by: sortBy,
          sort_order: sortOrder,
        },
      });

      return response;
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && activeTab === "request logs" && !hasBackendFilters,
    refetchInterval: isLiveTail && currentPage === 1 ? 15000 : false,
    placeholderData: keepPreviousData,
    refetchIntervalInBackground: true,
  });

  // Refetch when sort, page, or time range changes (backend filters use their own fetch, not the main query)
  useEffect(() => {
    if (hasBackendFilters && accessToken) {
      // Cancel any pending debounced search to prevent it from overwriting this page's results
      debouncedSearch.cancel();
      performSearch(filters, currentPage);
    }
    // Intentionally omitted from deps:
    // - `filters` / `debouncedSearch` / `performSearch`: filter changes are handled by
    //   handleFilterChange → debouncedSearch; adding them here would double-fetch on filter apply.
    // - `hasBackendFilters` / `accessToken`: stable across sort/page/time changes; including them
    //   would cause spurious re-runs when the filter state first becomes active.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortBy, sortOrder, currentPage, startTime, endTime, isCustomDate]);

  // Compute client-side filtered logs directly from query data and filters
  const spendLogsData = logsQuery.data;
  const clientDerivedFilteredLogs: PaginatedResponse = useMemo(() => {
    if (!spendLogsData) {
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
      return spendLogsData;
    }

    let filteredData = [...spendLogsData.data];

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
      total: spendLogsData.total,
      page: spendLogsData.page,
      page_size: spendLogsData.page_size,
      total_pages: spendLogsData.total_pages,
    };
  }, [spendLogsData, filters, hasBackendFilters]);

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

  return {
    logsQuery,
    filteredLogs,
    hasBackendFilters,
    allTeams,
    handleFilterChange,
    handleFilterReset,
  };
}
