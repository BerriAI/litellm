import moment from "moment";
import { useCallback, useEffect, useState, useRef, useMemo } from "react";
import { modelAvailableCall, uiSpendLogsCall } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { useQuery } from "@tanstack/react-query";
import { fetchAllKeyAliases, fetchAllTeams } from "../../components/key_team_helpers/filter_helpers";
import { debounce } from "lodash";
import { defaultPageSize } from "../constants";
import { PaginatedResponse } from ".";

export const FILTER_KEYS = {
  TEAM_ID: "Team ID",
  KEY_HASH: "Key Hash",
  REQUEST_ID: "Request ID",
  MODEL: "Model",
  USER_ID: "User ID",
  END_USER: "End User",
  STATUS: "Status",
  KEY_ALIAS: "Key Alias",
} as const;

export type FilterKey = keyof typeof FILTER_KEYS;
export type LogFilterState = Record<typeof FILTER_KEYS[FilterKey], string>;

export function useLogFilterLogic({
  logs,
  accessToken,
  startTime, // Receive from SpendLogsTable
  endTime,   // Receive from SpendLogsTable
  pageSize = defaultPageSize,
  isCustomDate,
  setCurrentPage,
  userID,  
  userRole 
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
}) {
  const defaultFilters = useMemo<LogFilterState>(() => ({
    [FILTER_KEYS.TEAM_ID]: "",
    [FILTER_KEYS.KEY_HASH]: "",
    [FILTER_KEYS.REQUEST_ID]: "",
    [FILTER_KEYS.MODEL]: "",
    [FILTER_KEYS.USER_ID]: "",
    [FILTER_KEYS.END_USER]: "",
    [FILTER_KEYS.STATUS]: "",
    [FILTER_KEYS.KEY_ALIAS]: ""
  }), []);

  const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
  const [filteredLogs, setFilteredLogs] = useState<PaginatedResponse>(logs);
  const lastSearchTimestamp = useRef(0);
  const performSearch = useCallback(async (filters: LogFilterState, page = 1) => {
    if (!accessToken) return;

    console.log("Filters being sent to API:", filters);
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
        page,
        pageSize,
        filters[FILTER_KEYS.USER_ID] || undefined,
        filters[FILTER_KEYS.END_USER] || undefined,
        filters[FILTER_KEYS.STATUS] || undefined,
        filters[FILTER_KEYS.MODEL] || undefined,
        filters[FILTER_KEYS.KEY_ALIAS] || undefined
      );

      if (currentTimestamp === lastSearchTimestamp.current && response.data) {
        setFilteredLogs(response);
      }
    } catch (error) {
      console.error("Error searching users:", error);
    }
  }, [accessToken, startTime, endTime, isCustomDate, pageSize]);

  const debouncedSearch = useMemo(
    () => debounce((filters: LogFilterState, page: number) => performSearch(filters, page), 300),
    [performSearch]
  );

  useEffect(() => {
    return () => debouncedSearch.cancel();
  }, [debouncedSearch]);

  const queryAllKeysQuery = useQuery({
    queryKey: ['allKeys'],
    queryFn: async () => {
      if (!accessToken) throw new Error('Access token required');
      return await fetchAllKeyAliases(accessToken);
    },
    enabled: !!accessToken
  });
  const allKeyAliases = queryAllKeysQuery.data || []

  // Apply filters to keys whenever logs or filters change
  useEffect(() => {
    if (!logs || !logs.data) {
      setFilteredLogs({
        data: [],
        total: 0,
        page: 1,
        page_size: 50,
        total_pages: 0
      });
      return;
    }

    // Only do client-side filtering if no backend filters are active
    const hasBackendFilters = 
      filters[FILTER_KEYS.KEY_ALIAS] || 
      filters[FILTER_KEYS.KEY_HASH] || 
      filters[FILTER_KEYS.REQUEST_ID] || 
      filters[FILTER_KEYS.USER_ID] || 
      filters[FILTER_KEYS.END_USER];

    if (hasBackendFilters) {
      // Backend is handling filtering, don't override the results
      return;
    }
  
    let filteredData = [...logs.data];
  
    if (filters[FILTER_KEYS.TEAM_ID]) {
      filteredData = filteredData.filter(
        log => log.team_id === filters[FILTER_KEYS.TEAM_ID]
      );
    }

    if (filters[FILTER_KEYS.STATUS]) {
      filteredData = filteredData.filter(
        log => {
          if (filters[FILTER_KEYS.STATUS] === 'success') {
            return !log.status || log.status === 'success';
          }
          return log.status === filters[FILTER_KEYS.STATUS];
        }
      );
    }

    if (filters[FILTER_KEYS.MODEL]) {
      filteredData = filteredData.filter(
        log => log.model === filters[FILTER_KEYS.MODEL]
      );
    }
    
    if (filters[FILTER_KEYS.KEY_HASH]) {
      filteredData = filteredData.filter(
        log => log.api_key === filters[FILTER_KEYS.KEY_HASH]
      );
    }

    if (filters[FILTER_KEYS.END_USER]) {
      filteredData = filteredData.filter(
        log => log.end_user === filters[FILTER_KEYS.END_USER]
      );
    }
    
    const newFilteredLogs: PaginatedResponse = {
      data: filteredData,
      total: logs.total,
      page: logs.page,
      page_size: logs.page_size,
      total_pages: logs.total_pages,
    };
  
    if (JSON.stringify(newFilteredLogs) !== JSON.stringify(filteredLogs)) {
      setFilteredLogs(newFilteredLogs);
    }
  }, [logs, filters, filteredLogs, accessToken]);

  

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
        setCurrentPage(1);
        debouncedSearch(updatedFilters, 1);
      }
      
      return updatedFilters as LogFilterState;
    });
  };
    

  const handleFilterReset = () => {
    // Reset filters state
    setFilters(defaultFilters);
    
    // Reset selections
    debouncedSearch(defaultFilters, 1);
  };

  return {
    filters,
    filteredLogs,
    allKeyAliases,
    allTeams,
    handleFilterChange,
    handleFilterReset,
  };
}
