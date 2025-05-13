import moment from "moment";
import { useCallback, useEffect, useState, useRef } from "react";
import { KeyResponse } from "../key_team_helpers/key_list";
import { keyListCall, Organization, uiSpendLogsCall } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { fetchAllKeyAliases, fetchAllTeams } from "../../components/key_team_helpers/filter_helpers";
import { Setter } from "@/types";
import { debounce } from "lodash";
import { defaultPageSize } from "../constants";
import { LogEntry } from "./columns";

export interface LogFilterState {
  'Team ID': string;
  // 'Key Alias': string;
  'Key Hash': string;
  'Request ID': string;
  'Model': string;
  'User ID': string; 
  // 'Cache Hit': 'true' | 'false' | ''; // Example for boolean filter
  // 'Status': 'success' | 'failure' | ''; // Example for status filter
}

// interface PaginatedResponse {
//   data: LogEntry[];
//   total: number;
//   page: number;
//   page_size: number;
//   total_pages: number;
// }

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
  const defaultFilters: LogFilterState = {
    'Team ID': '',
    // 'Key Alias': '',
    'Key Hash': '',
    'Request ID': '',
    'Model': '',
    'User ID': '',
    // 'Cache Hit': '',
    // 'Status': '',
    // ...initialFilters
  }
  const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>(logs);
  const lastSearchTimestamp = useRef(0);
  const debouncedSearch = useCallback(
    debounce(async (filters: LogFilterState) => {
      if (!accessToken) {
        return;
      }
  
      const currentTimestamp = Date.now();
      lastSearchTimestamp.current = currentTimestamp;

      const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = isCustomDate 
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      const apiKeyParam = filters["Key Hash"] || undefined;
      const teamIdParam = filters["Team ID"] || undefined;
      const requestIdParam = filters["Request ID"] || undefined;
      const userIdParam = filters["User ID"] || undefined;
  
      try {
        // Make the API call using userListCall with all filter parameters
        const response = await uiSpendLogsCall(
          accessToken,
          apiKeyParam,
          teamIdParam,
          requestIdParam,
          formattedStartTime,
          formattedEndTime,
          1,
          pageSize,
          userIdParam
          // If uiSpendLogsCall is updated to accept more params (e.g., model), pass them here:
          // modelParam, 
        );
        // Only update state if this is the most recent search
        if (currentTimestamp === lastSearchTimestamp.current) {
          if (response.data) {
            setFilteredLogs(response.data);
            console.log("called from debouncedSearch filters:", JSON.stringify(filters));
            console.log("called from debouncedSearch data:", JSON.stringify(response.data));
          }
        }
      } catch (error) {
        console.error("Error searching users:", error);
      }
    }, 300),
    [accessToken]
  );
  // Apply filters to keys whenever keys or filters change
  useEffect(() => {
    if (!logs) {
      setFilteredLogs([]);
      return;
    }
  
    let result = [...logs];
  
    if (filters['Team ID']) {
      result = result.filter(log => log.team_id === filters['Team ID']);
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

  // const handleFilterChange = (newFilters: Record<string, string>) => {
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
