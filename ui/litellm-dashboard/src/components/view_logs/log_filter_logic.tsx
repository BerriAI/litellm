import moment from "moment";
import { useEffect, useState } from "react";
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
}

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

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
  // Debounce filters so text inputs (Key Hash, Error Message) don't fire a
  // request per keystroke. Dropdown selects get a 300ms delay too, which is
  // imperceptible since the user just clicked an option.
  const debouncedFilters = useDebouncedValue(filters, 300);

  const logsQuery = useQuery<PaginatedResponse>({
    queryKey: [
      "logs",
      "table",
      currentPage,
      pageSize,
      startTime,
      endTime,
      debouncedFilters,
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
          api_key: debouncedFilters[FILTER_KEYS.KEY_HASH] || undefined,
          team_id: debouncedFilters[FILTER_KEYS.TEAM_ID] || undefined,
          request_id: debouncedFilters[FILTER_KEYS.REQUEST_ID] || undefined,
          user_id: debouncedFilters[FILTER_KEYS.USER_ID] || (filterByCurrentUser ? userID ?? undefined : undefined),
          end_user: debouncedFilters[FILTER_KEYS.END_USER] || undefined,
          status_filter: debouncedFilters[FILTER_KEYS.STATUS] || undefined,
          model_id: debouncedFilters[FILTER_KEYS.MODEL] || undefined,
          key_alias: debouncedFilters[FILTER_KEYS.KEY_ALIAS] || undefined,
          error_code: debouncedFilters[FILTER_KEYS.ERROR_CODE] || undefined,
          error_message: debouncedFilters[FILTER_KEYS.ERROR_MESSAGE] || undefined,
          sort_by: sortBy,
          sort_order: sortOrder,
        },
      });

      return response;
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && activeTab === "request logs",
    refetchInterval: isLiveTail && currentPage === 1 ? 15000 : false,
    placeholderData: keepPreviousData,
    refetchIntervalInBackground: true,
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
