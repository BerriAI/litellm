import moment from "moment";
import { useCallback, useEffect, useState, useRef, useMemo } from "react";
import { keyListCall, uiSpendLogsCall } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { useQuery } from "@tanstack/react-query";
import { fetchAllKeyAliases, fetchAllTeams } from "../../components/key_team_helpers/filter_helpers";
import { debounce } from "lodash";
import { defaultPageSize } from "../constants";
import { LogEntry } from "./columns";

export const FILTER_KEYS = {
  TEAM_ID: "Team ID",
  KEY_HASH: "Key Hash",
  REQUEST_ID: "Request ID",
  MODEL: "Model",
  USER_ID: "User ID"
} as const;

export type FilterKey = keyof typeof FILTER_KEYS;
export type LogFilterState = Record<typeof FILTER_KEYS[FilterKey], string>;

export function useLogFilterLogic({
  logs,
  accessToken,
  startTime, // Receive from SpendLogsTable
  endTime,   // Receive from SpendLogsTable
  pageSize = defaultPageSize,
  initialPage = 1,
  initialFilters = {},
  isCustomDate,
  currentPage
}: {
  logs: LogEntry[];
  accessToken: string | null;
  startTime: string;
  endTime: string;
  pageSize?: number;
  initialPage?: number;
  initialFilters?: Partial<LogFilterState>;
  isCustomDate: boolean;
  currentPage: number;
}) {
  const defaultFilters = useMemo<LogFilterState>(() => ({
    [FILTER_KEYS.TEAM_ID]: "",
    [FILTER_KEYS.KEY_HASH]: "",
    [FILTER_KEYS.REQUEST_ID]: "",
    [FILTER_KEYS.MODEL]: "",
    [FILTER_KEYS.USER_ID]: "",
  }), []);

  const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>(logs);
  const lastSearchTimestamp = useRef(0);
  const performSearch = useCallback(async (filters: LogFilterState) => {
    if (!accessToken) return;

    const currentTimestamp = Date.now();
    lastSearchTimestamp.current = currentTimestamp;

    const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
    const formattedEndTime = isCustomDate
      ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
      : moment().utc().format("YYYY-MM-DD HH:mm:ss");

    try {
      const response = await uiSpendLogsCall(
        accessToken,
        filters[FILTER_KEYS.KEY_HASH] || undefined,
        filters[FILTER_KEYS.TEAM_ID] || undefined,
        filters[FILTER_KEYS.REQUEST_ID] || undefined,
        formattedStartTime,
        formattedEndTime,
        currentPage,
        pageSize,
        filters[FILTER_KEYS.USER_ID] || undefined
      );

      if (currentTimestamp === lastSearchTimestamp.current && response.data) {
        setFilteredLogs(response.data);
        console.log("called from debouncedSearch filters:", JSON.stringify(filters));
        console.log("called from debouncedSearch data:", JSON.stringify(response.data));
      }
    } catch (error) {
      console.error("Error searching users:", error);
    }
  }, [accessToken]);

  const debouncedSearch = useMemo(() => debounce(performSearch, 300), [performSearch]);

  useEffect(() => {
    return () => debouncedSearch.cancel();
  }, [debouncedSearch]);

  // Apply filters to keys whenever keys or filters change
  useEffect(() => {
    if (!logs) {
      setFilteredLogs([]);
      return;
    }
  
    let result = [...logs];
  
    if (filters[FILTER_KEYS.TEAM_ID]) {
      result = result.filter(log => log.team_id === filters[FILTER_KEYS.TEAM_ID]);
    }
  
    // Only update state if the filteredLogs actually changed
    if (JSON.stringify(result) !== JSON.stringify(filteredLogs)) {
      setFilteredLogs(result);
    }
  }, [logs, filters]);

  const queryAllKeysQuery = useQuery({
    queryKey: ['allKeys'],
    queryFn: async () => {
      if (!accessToken) throw new Error('Access token required');
      return await fetchAllKeyAliases(accessToken);
    },
    enabled: !!accessToken
  });
  const allKeyAliases = queryAllKeysQuery.data || []

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
    setFilters(prev => {
      const updatedFilters = { ...prev, ...newFilters };
      
      // Ensure all keys in LogFilterState are present, defaulting to '' if not in newFilters
      for (const key of Object.keys(defaultFilters) as Array<keyof LogFilterState>) {
        if (!(key in updatedFilters)) {
          updatedFilters[key] = defaultFilters[key];
        }
      }
      
      // Only call debouncedSearch if filters have actually changed
      if (JSON.stringify(updatedFilters) !== JSON.stringify(prev)) {
        debouncedSearch(updatedFilters);
      }
      
      return updatedFilters as LogFilterState;
    });
  };
    

  const handleFilterReset = () => {
    // Reset filters state
    setFilters(defaultFilters);
    
    // Reset selections
    debouncedSearch(defaultFilters);
  };

  return {
    filters,
    filteredLogs,
    allKeyAliases,
    allTeams,
    handleFilterChange,
    handleFilterReset
  };
}
