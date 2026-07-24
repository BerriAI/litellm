import moment from "moment";
import { keepPreviousData, useQuery, type UseQueryOptions } from "@tanstack/react-query";
import type { ColumnFiltersState, PaginationState, SortingState } from "@tanstack/react-table";
import { uiSpendLogsCall } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { fetchAllTeams } from "../../components/key_team_helpers/filter_helpers";
import { defaultPageSize } from "../constants";
import { LOGS_SORT_FIELD_MAP, type LogEntry, type LogsSortField } from "./columns";

export interface PaginatedResponse {
  data: LogEntry[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  total_is_capped?: boolean;
}

export const LOG_FILTER_IDS = {
  TEAM_ID: "team_id",
  STATUS: "status",
  KEY_ALIAS: "key_alias",
  END_USER: "end_user",
  ERROR_CODE: "error_code",
  ERROR_MESSAGE: "error_message",
  KEY_HASH: "key_hash",
  SESSION_ID: "session_id",
  MODEL_ID: "model_id",
  PUBLIC_MODEL_OR_SEARCH_TOOL: "model",
  REQUEST_ID: "request_id",
  USER_ID: "user_id",
} as const;

export const LOG_FILTER_LABELS: Record<string, string> = {
  [LOG_FILTER_IDS.TEAM_ID]: "Team ID",
  [LOG_FILTER_IDS.STATUS]: "Status",
  [LOG_FILTER_IDS.KEY_ALIAS]: "Key Alias",
  [LOG_FILTER_IDS.END_USER]: "End User",
  [LOG_FILTER_IDS.ERROR_CODE]: "Error Code",
  [LOG_FILTER_IDS.ERROR_MESSAGE]: "Error Message",
  [LOG_FILTER_IDS.KEY_HASH]: "Key Hash",
  [LOG_FILTER_IDS.SESSION_ID]: "Session ID",
  [LOG_FILTER_IDS.MODEL_ID]: "Model",
  [LOG_FILTER_IDS.PUBLIC_MODEL_OR_SEARCH_TOOL]: "Public model / search tool",
};

export const LIVE_TAIL_INTERVAL_MS = 15000;

export const getLiveTailRefetchInterval = (isLiveTail: boolean, pageIndex: number): number | false =>
  isLiveTail && pageIndex === 0 ? LIVE_TAIL_INTERVAL_MS : false;

export const DEFAULT_LOGS_SORTING: SortingState = [{ id: "startTime", desc: true }];

const isSortField = (id: string): id is LogsSortField => Object.hasOwn(LOGS_SORT_FIELD_MAP, id);

export const getFilterValue = (columnFilters: ColumnFiltersState, columnId: string): string | undefined => {
  const entry = columnFilters.find((filter) => filter.id === columnId);
  if (typeof entry?.value !== "string") return undefined;
  const trimmed = entry.value.trim();
  return trimmed === "" ? undefined : trimmed;
};

export function useLogFilterLogic({
  accessToken,
  token,
  userRole,
  userID,
  columnFilters,
  filterByCurrentUser,
  activeTab,
  isLiveTail,
  startTime,
  endTime,
  pagination,
  isCustomDate,
  sorting,
}: {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  columnFilters: ColumnFiltersState;
  filterByCurrentUser: boolean | null;
  activeTab: string;
  isLiveTail: boolean;
  startTime: string;
  endTime: string;
  pagination: PaginationState;
  isCustomDate: boolean;
  sorting: SortingState;
}) {
  const pageSize = pagination.pageSize || defaultPageSize;
  const activeSort = sorting[0] ?? DEFAULT_LOGS_SORTING[0];
  const sortBy: LogsSortField = isSortField(activeSort.id) ? activeSort.id : "startTime";
  const sortOrder: "asc" | "desc" = activeSort.desc ? "desc" : "asc";

  const logsQueryOptions: UseQueryOptions<PaginatedResponse> = {
    queryKey: [
      "logs",
      "table",
      pagination.pageIndex,
      pageSize,
      startTime,
      endTime,
      isCustomDate,
      columnFilters,
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

      const userIdFilter = getFilterValue(columnFilters, LOG_FILTER_IDS.USER_ID);

      return await uiSpendLogsCall({
        accessToken,
        start_date: formattedStartTime,
        end_date: formattedEndTime,
        page: pagination.pageIndex + 1,
        page_size: pageSize,
        params: {
          api_key: getFilterValue(columnFilters, LOG_FILTER_IDS.KEY_HASH),
          team_id: getFilterValue(columnFilters, LOG_FILTER_IDS.TEAM_ID),
          request_id: getFilterValue(columnFilters, LOG_FILTER_IDS.REQUEST_ID),
          session_id: getFilterValue(columnFilters, LOG_FILTER_IDS.SESSION_ID),
          user_id: userIdFilter ?? (filterByCurrentUser ? userID ?? undefined : undefined),
          end_user: getFilterValue(columnFilters, LOG_FILTER_IDS.END_USER),
          status_filter: getFilterValue(columnFilters, LOG_FILTER_IDS.STATUS),
          model_id: getFilterValue(columnFilters, LOG_FILTER_IDS.MODEL_ID),
          model: getFilterValue(columnFilters, LOG_FILTER_IDS.PUBLIC_MODEL_OR_SEARCH_TOOL),
          key_alias: getFilterValue(columnFilters, LOG_FILTER_IDS.KEY_ALIAS),
          error_code: getFilterValue(columnFilters, LOG_FILTER_IDS.ERROR_CODE),
          error_message: getFilterValue(columnFilters, LOG_FILTER_IDS.ERROR_MESSAGE),
          sort_by: sortBy,
          sort_order: sortOrder,
        },
      });
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && activeTab === "request logs",
    refetchInterval: getLiveTailRefetchInterval(isLiveTail, pagination.pageIndex),
    placeholderData: keepPreviousData,
    refetchIntervalInBackground: false,
  };

  const logsQuery = useQuery(logsQueryOptions);

  const filteredLogs: PaginatedResponse = logsQuery.data ?? {
    data: [],
    total: 0,
    page: 1,
    page_size: pageSize,
    total_pages: 0,
  };

  const allTeamsQueryOptions: UseQueryOptions<Team[], Error> = {
    queryKey: ["allTeamsForLogFilters", accessToken],
    queryFn: async () => {
      if (!accessToken) return [];
      const teamsData = await fetchAllTeams(accessToken);
      return teamsData || [];
    },
    enabled: !!accessToken,
  };

  const { data: allTeams } = useQuery(allTeamsQueryOptions);

  return {
    logsQuery,
    filteredLogs,
    allTeams,
  };
}
