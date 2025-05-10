import { useCallback, useEffect, useState, useRef, useMemo } from "react";
import { LogEntry } from "./columns"; 
import { uiSpendLogsCall, Organization, Team, UserInfo, teamListCall, userListCall, keyListCall as fetchAllKeysCall, modelAvailableCall } from "../networking"; 
import { KeyResponse } from "../key_team_helpers/key_list";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Setter } from "@/types";
import { debounce } from "lodash";
import { defaultPageSize } from "../constants";
import moment from "moment";
import { fetchAllKeyAliases, fetchAllTeams } from "../key_team_helpers/filter_helpers"; // Import fetchAllTeams

// Define the shape of the filter state for logs
export interface LogFilterState {
  'Team ID': string;
  'Organization ID': string; // Added for potential future use
  'Key Alias': string;
  'Key Hash': string;
  'Request ID': string;
  'Model': string;
  'User': string; 
  'Cache Hit': 'true' | 'false' | ''; // Example for boolean filter
  'Status': 'success' | 'failure' | ''; // Example for status filter
  [key: string]: string | ''; // Allow for other string-based filters
}

// Define the shape of the pagination state
export interface PaginationState {
  currentPage: number;
  totalPages: number;
  totalCount: number;
  pageSize: number;
}

export function useLogFilterLogic({
  accessToken,
  startTime, // Receive from SpendLogsTable
  endTime,   // Receive from SpendLogsTable
  pageSize = defaultPageSize,
  initialPage = 1,
  initialFilters = {},
  userID,  
  userRole 
}: {
  accessToken: string | null;
  startTime: string;
  endTime: string;
  pageSize?: number;
  initialPage?: number;
  initialFilters?: Partial<LogFilterState>;
  userID: string | null; 
  userRole: string | null; 
}) {
  const defaultFilters: LogFilterState = {
    'Team ID': '',
    'Organization ID': '',
    'Key Alias': '',
    'Key Hash': '',
    'Request ID': '',
    'Model': '',
    'User': '',
    'Cache Hit': '',
    'Status': '',
    ...initialFilters
  };

  const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
  const [currentPage, setCurrentPage] = useState<number>(initialPage);
  const queryClient = useQueryClient();
  
  // States for manually managed data, loading, and error
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [paginationDetails, setPaginationDetails] = useState<PaginationState>({ 
    currentPage: initialPage, 
    totalPages: 0, 
    totalCount: 0, 
    pageSize 
  });
  const [isLoadingLogs, setIsLoadingLogs] = useState<boolean>(false);
  const [logsError, setLogsError] = useState<Error | null>(null);
  const lastSearchTimestamp = useRef(0);

  const queryAllKeysQuery = useQuery({
    queryKey: ['allKeysForLogFilters', accessToken], // Ensure unique queryKey
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

  const { data: allUsers } = useQuery<UserInfo[], Error>({
    queryKey: ["allUsersForLogFilters", accessToken],
    queryFn: async () => {
      if (!accessToken) return [];
      // Assuming userListCall fetches all users. Adapt if pagination or specific params are needed.
      const response = await userListCall(accessToken, null, 1, 100); // Fetch a large number for dropdown
      return response.users || [];
    },
    enabled: !!accessToken,
  });

  const { data: allModels = [] } = useQuery<string[], Error>({
    queryKey: ['allModels', accessToken, userID, userRole],
    queryFn: async () => {
      if (!accessToken || !userID || !userRole) return [];

      const response = await modelAvailableCall(
        accessToken,
        userID,
        userRole,
        false, // return_wildcard_routes
        null // teamID
      );

      return response.data.map((model: { id: string }) => model.id);
    },
    enabled: !!accessToken && !!userID && !!userRole,
  });
  
  // Debounced API call
  const debouncedSearch = useCallback(
    debounce(async (currentFilters: LogFilterState, pageToFetch: number) => {
      if (!accessToken) {
        setLogEntries([]);
        setPaginationDetails({ currentPage: pageToFetch, totalPages: 0, totalCount: 0, pageSize });
        setIsLoadingLogs(false);
        setLogsError(null);
        return;
      }

      const currentTimestamp = Date.now();
      lastSearchTimestamp.current = currentTimestamp;
      setIsLoadingLogs(true);
      setLogsError(null);

      try {
        const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
        const formattedEndTime = moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss");

        const apiKeyParam = currentFilters['Key Hash'] || undefined;
        const teamIdParam = currentFilters['Team ID'] || undefined;
        const requestIdParam = currentFilters['Request ID'] || undefined;
        const userIdParam = currentFilters['User'] || undefined;
        const statusParam = currentFilters['Status'] || undefined;
        const modelParam = currentFilters['Model'] || undefined;

        const response = await uiSpendLogsCall(
          accessToken,
          apiKeyParam,
          teamIdParam,
          requestIdParam,
          formattedStartTime,
          formattedEndTime,
          pageToFetch,
          pageSize,
          userIdParam,
          statusParam, 
          modelParam, 
        );

        if (currentTimestamp === lastSearchTimestamp.current) {
          if (response && response.data) {
            setLogEntries(response.data);
            setPaginationDetails({
              currentPage: response.page,
              totalPages: response.total_pages,
              totalCount: response.total,
              pageSize: response.page_size || pageSize,
            });
          } else {
            setLogEntries([]);
            setPaginationDetails({ currentPage: pageToFetch, totalPages: 0, totalCount: 0, pageSize });
          }
        }
      } catch (error: any) {
        if (currentTimestamp === lastSearchTimestamp.current) {
          console.error("Error fetching logs:", error);
          setLogsError(error);
          setLogEntries([]);
          setPaginationDetails({ currentPage: pageToFetch, totalPages: 0, totalCount: 0, pageSize });
        }
      } finally {
        if (currentTimestamp === lastSearchTimestamp.current) {
          setIsLoadingLogs(false);
        }
      }
    }, 300),
    [accessToken, startTime, endTime, pageSize, setLogEntries, setPaginationDetails, setIsLoadingLogs, setLogsError]
  );

  // useEffect to trigger debouncedSearch when dependencies change
  useEffect(() => {
    if (accessToken) {
      debouncedSearch(filters, currentPage);
    } else {
      // Clear data if accessToken is lost or not present
      setLogEntries([]);
      setPaginationDetails(prev => ({ ...prev, currentPage, totalPages: 0, totalCount: 0 }));
      setIsLoadingLogs(false);
      setLogsError(null);
    }
    // Cleanup function for debounce
    return () => {
      debouncedSearch.cancel();
    };
  }, [accessToken, filters, currentPage, startTime, endTime, pageSize, debouncedSearch]);


  const handleFilterChange = (newFilters: Partial<LogFilterState>) => {
    setFilters(prev => {
      const updatedFilters = { ...prev, ...newFilters }; // Simpler update
      // Ensure all keys in LogFilterState are present, defaulting to '' if not in newFilters
      for (const key of Object.keys(defaultFilters) as Array<keyof LogFilterState>) {
        if (!(key in updatedFilters)) {
          updatedFilters[key] = defaultFilters[key];
        }
      }
      return updatedFilters as LogFilterState;
    });
    setCurrentPage(1); // Reset to first page when filters change
    // debouncedSearch will be called by the useEffect hook
  };

  const handleFilterReset = () => {
    setFilters(defaultFilters);
    setCurrentPage(1);
    // debouncedSearch will be called by the useEffect hook
  };

  const pagination: PaginationState = {
    currentPage: paginationDetails.currentPage,
    totalPages: paginationDetails.totalPages,
    totalCount: paginationDetails.totalCount,
    pageSize: paginationDetails.pageSize,
  };

  const handleRefresh = () => {
    // Reset to first page
    setCurrentPage(1);
    
    // Invalidate and refetch all relevant queries
    queryClient.invalidateQueries({ queryKey: ['allKeysForLogFilters'] });
    queryClient.invalidateQueries({ queryKey: ['allTeamsForLogFilters'] });
    queryClient.invalidateQueries({ queryKey: ['allUsersForLogFilters'] });
    queryClient.invalidateQueries({ queryKey: ['allModels'] });
    
    // Force an immediate refetch of the logs data
    debouncedSearch(filters, 1);
  };

  return {
    filters,
    filteredLogs: logEntries,
    allKeyAliases,
    allTeams: allTeams || [],
    allUsers: allUsers || [],
    allOrganizations: [], // Placeholder for now, can be fetched if needed
    allModels: allModels || [],
    handleFilterChange,
    handleFilterReset,
    isLoading: isLoadingLogs,
    pagination,
    setCurrentPage,
    error: logsError,
    handleRefresh
  };
} 