import moment from "moment";
import { useEffect, useMemo, useState } from "react";
import { uiSpendLogsCall } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { fetchAllTeams } from "../../components/key_team_helpers/filter_helpers";
import { defaultPageSize } from "../constants";
import type { LogEntry, LogsSortField } from "./columns";

export interface PaginatedResponse {
  data: LogEntry[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  total_is_capped?: boolean;
}

function useDebouncedValue<T>(value: T, delayMs: number): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return [debounced, setDebounced];
}

/** Spend log `model` column (LLM public model name or `search_tool_name` for /search). */
export const FILTER_KEYS = {
  TEAM_ID: "Team ID",
  KEY_HASH: "Key Hash",
  REQUEST_ID: "Request ID",
  SESSION_ID: "Session ID",
  MODEL: "Model",
  /** Exact match on LiteLLM_SpendLogs.model — use for search tools and public model names. */
  PUBLIC_MODEL_OR_SEARCH_TOOL: "Public model / search tool",
  USER_ID: "User ID",
  END_USER: "End User",
  STATUS: "Status",
  KEY_ALIAS: "Key Alias",
  ERROR_CODE: "Error Code",
  ERROR_MESSAGE: "Error Message",
} as const;

export type FilterKey = keyof typeof FILTER_KEYS;
export type LogFilterState = Record<(typeof FILTER_KEYS)[FilterKey], string>;

// Keys whose UI is a free-form text input; only these need debouncing.
const TEXT_FILTER_KEYS: readonly (keyof LogFilterState)[] = [
  FILTER_KEYS.KEY_HASH,
  FILTER_KEYS.ERROR_MESSAGE,
  FILTER_KEYS.REQUEST_ID,
  FILTER_KEYS.SESSION_ID,
  FILTER_KEYS.USER_ID,
  FILTER_KEYS.PUBLIC_MODEL_OR_SEARCH_TOOL,
];

// Live-tail polls every 15s, but only on page 1 (newest) while live tail is on.
export const LIVE_TAIL_INTERVAL_MS = 15000;
export const getLiveTailRefetchInterval = (isLiveTail: boolean, currentPage: number): number | false =>
  isLiveTail && currentPage === 1 ? LIVE_TAIL_INTERVAL_MS : false;

export const defaultFilters: LogFilterState = {
  [FILTER_KEYS.TEAM_ID]: "",
  [FILTER_KEYS.KEY_HASH]: "",
  [FILTER_KEYS.REQUEST_ID]: "",
  [FILTER_KEYS.SESSION_ID]: "",
  [FILTER_KEYS.MODEL]: "",
  [FILTER_KEYS.PUBLIC_MODEL_OR_SEARCH_TOOL]: "",
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
  const [debouncedFilters, setDebouncedFilters] = useDebouncedValue(filters, 300);

  // Live values for dropdown keys, debounced for text keys.
  const effectiveFilters = useMemo(() => {
    const merged = { ...filters };
    for (const k of TEXT_FILTER_KEYS) {
      merged[k] = debouncedFilters[k];
    }
    return merged;
  }, [filters, debouncedFilters]);

  const logsQuery = useQuery<PaginatedResponse>({
    queryKey: [
      "logs",
      "table",
      currentPage,
      pageSize,
      startTime,
      endTime,
      isCustomDate,
      effectiveFilters,
      filterByCurrentUser ? userID : null,
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
          api_key: effectiveFilters[FILTER_KEYS.KEY_HASH] || undefined,
          team_id: effectiveFilters[FILTER_KEYS.TEAM_ID] || undefined,
          request_id: effectiveFilters[FILTER_KEYS.REQUEST_ID] || undefined,
          session_id: effectiveFilters[FILTER_KEYS.SESSION_ID] || undefined,
          user_id: effectiveFilters[FILTER_KEYS.USER_ID] || (filterByCurrentUser ? userID ?? undefined : undefined),
          end_user: effectiveFilters[FILTER_KEYS.END_USER] || undefined,
          status_filter: effectiveFilters[FILTER_KEYS.STATUS] || undefined,
          model_id: effectiveFilters[FILTER_KEYS.MODEL] || undefined,
          model: effectiveFilters[FILTER_KEYS.PUBLIC_MODEL_OR_SEARCH_TOOL] || undefined,
          key_alias: effectiveFilters[FILTER_KEYS.KEY_ALIAS] || undefined,
          error_code: effectiveFilters[FILTER_KEYS.ERROR_CODE] || undefined,
          error_message: effectiveFilters[FILTER_KEYS.ERROR_MESSAGE] || undefined,
          sort_by: sortBy,
          sort_order: sortOrder,
        },
      });

      return response;
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && activeTab === "request logs",
    refetchInterval: getLiveTailRefetchInterval(isLiveTail, currentPage),
    placeholderData: keepPreviousData,
    // Only live-tail-poll while the tab is visible.
    refetchIntervalInBackground: false,
  });

  const filteredLogs: PaginatedResponse = logsQuery.data ?? {
    data: [],
    total: 0,
    page: 1,
    page_size: pageSize,
    total_pages: 0,
  };

  const { data: allTeams } = useQuery<Team[], Error>({
    queryKey: ["allTeamsForLogFilters", accessToken],
    queryFn: async () => {
      if (!accessToken) return [];
      const teamsData = await fetchAllTeams(accessToken);
      return teamsData || [];
    },
    enabled: !!accessToken,
  });

  const handleFilterChange = (newFilters: Partial<LogFilterState>) => {
    setFilters((prev) => {
      const updatedFilters = { ...prev, ...newFilters };
      for (const key of Object.keys(defaultFilters) as Array<keyof LogFilterState>) {
        if (!(key in updatedFilters)) {
          updatedFilters[key] = defaultFilters[key];
        }
      }
      if (JSON.stringify(updatedFilters) !== JSON.stringify(prev)) {
        setCurrentPage(1);
      }
      return updatedFilters as LogFilterState;
    });
  };

  const handleFilterReset = () => {
    setFilters(defaultFilters);
    setDebouncedFilters(defaultFilters);
    setCurrentPage(1);
  };

  return {
    logsQuery,
    filteredLogs,
    allTeams,
    handleFilterChange,
    handleFilterReset,
  };
}
