import { useCallback, useEffect, useState, useRef, useMemo } from "react";
import { LogEntry } from "./columns"; 
import { uiSpendLogsCall, Organization, Team, UserInfo, teamListCall, userListCall, keyListCall as fetchAllKeysCall, modelAvailableCall } from "../networking"; 
import { KeyResponse } from "../key_team_helpers/key_list";
import { useQuery, useQueryClient, QueryKey } from "@tanstack/react-query";
import { Setter } from "@/types";
// import { debounce } from "lodash"; // No longer needed
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

// Define the expected response structure from uiSpendLogsCall for type safety
interface SpendLogsResponse {
  data: LogEntry[];
  page: number;
  total_pages: number;
  total: number;
  page_size: number;
}


export function useLogFilterLogic({
  accessToken,
  startTime, // Receive from SpendLogsTable
  endTime,   // Receive from SpendLogsTable
  pageSize = defaultPageSize,
  initialPage = 1,
  initialFilters = {},
  userID,  
  userRole,
  autoRefreshInterval // Added for auto-refresh
}: {
  accessToken: string | null;
  startTime: string;
  endTime: string;
  pageSize?: number;
  initialPage?: number;
  initialFilters?: Partial<LogFilterState>;
  userID: string | null; 
  userRole: string | null; 
  autoRefreshInterval?: number; // Added for auto-refresh
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
  
  const queryAllKeysQuery = useQuery<string[], Error>({
    queryKey: ['allKeysForLogFilters', accessToken], 
    queryFn: async () => {
      if (!accessToken) throw new Error('Access token required');
      return await fetchAllKeyAliases(accessToken);
    },
    enabled: !!accessToken
  });
  const allKeyAliases = queryAllKeysQuery.data || [];

  const { data: allTeams } = useQuery<Team[], Error>({
    queryKey: ["allTeamsForLogFilters", accessToken],
    queryFn: async () => {
      if (!accessToken) return [];
      const teamsData = await fetchAllTeams(accessToken);
      return teamsData || []; 
    },
    enabled: !!accessToken,
  });

  const { data: allUsers } = useQuery<UserInfo[], Error>({
    queryKey: ["allUsersForLogFilters", accessToken],
    queryFn: async () => {
      if (!accessToken) return [];
      const response = await userListCall(accessToken, null, 1, 100); 
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
        false, 
        null 
      );
      return response.data.map((model: { id: string }) => model.id);
    },
    enabled: !!accessToken && !!userID && !!userRole,
  });
  
  const logsQueryKey: QueryKey = ['spendLogs', accessToken, filters, currentPage, startTime, endTime, pageSize, userID, userRole];

  const logsQueryFn = async (): Promise<SpendLogsResponse> => {
    const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
    const formattedEndTime = moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss");

    const apiKeyParam = filters['Key Hash'] || undefined;
    const keyAliasParam = filters['Key Alias'] || undefined;
    const teamIdParam = filters['Team ID'] || undefined;
    const requestIdParam = filters['Request ID'] || undefined;
    const userIdParamFilter = filters['User'] || undefined; 
    const statusParam = filters['Status'] || undefined;
    const modelParam = filters['Model'] || undefined;

    if (!accessToken) {
      // This case should ideally be prevented by the `enabled` option
      // Returning a structure that matches SpendLogsResponse to satisfy Promise type
      return { data: [], page: 1, total_pages: 0, total: 0, page_size: pageSize }; 
    }
    
    const response = await uiSpendLogsCall(
      accessToken,
      apiKeyParam,
      teamIdParam,
      requestIdParam,
      formattedStartTime,
      formattedEndTime,
      currentPage,
      pageSize,
      userIdParamFilter,
      statusParam, 
      modelParam, 
    );
    // Assuming uiSpendLogsCall returns a type compatible with SpendLogsResponse
    return response as SpendLogsResponse;
  };

  const logsQuery = useQuery<SpendLogsResponse, Error, SpendLogsResponse, QueryKey>({
    queryKey: logsQueryKey, 
    queryFn: logsQueryFn, 
    enabled: !!accessToken && !!userID && !!userRole, 
    refetchInterval: autoRefreshInterval && autoRefreshInterval > 0 ? autoRefreshInterval : undefined,
    refetchIntervalInBackground: true,
    placeholderData: (previousData) => previousData,
    staleTime: 0,
    gcTime: 0,
    retry: 1,
    notifyOnChangeProps: ['data']
  });

  // Handle success/error cases in useEffect
  useEffect(() => {
    if (logsQuery.error) {
      console.error('Error fetching logs:', logsQuery.error);
      setCurrentPage(1);
    }
    if (logsQuery.data && logsQuery.data.total_pages > 0 && currentPage > logsQuery.data.total_pages) {
      setCurrentPage(1);
    }
  }, [logsQuery.error, logsQuery.data, currentPage]);

  const handleFilterChange = (newFilters: Partial<LogFilterState>) => {
    setFilters(prev => {
      const updatedFilters = { ...prev, ...newFilters }; 
      for (const key of Object.keys(defaultFilters) as Array<keyof LogFilterState>) {
        if (!(key in updatedFilters)) {
          updatedFilters[key] = defaultFilters[key];
        }
      }
      return updatedFilters as LogFilterState;
    });
    setCurrentPage(1); 
  };

  const handleFilterReset = () => {
    setFilters(defaultFilters);
    setCurrentPage(1);
  };
  
  const pagination: PaginationState = useMemo(() => {
    const data: SpendLogsResponse | undefined = logsQuery.data;
    return {
        currentPage: currentPage,
        totalPages: data?.total_pages ?? 0,
        totalCount: data?.total ?? 0,
        pageSize: data?.page_size ?? pageSize,
    };
  }, [logsQuery.data, currentPage, pageSize]);

  const [isManualRefreshing, setIsManualRefreshing] = useState(false);

  const handleRefresh = () => {
    setIsManualRefreshing(true); // <-- ADD THIS LINE
    // First invalidate all the filter queries
    queryClient.invalidateQueries({ queryKey: ['allKeysForLogFilters'] });
    queryClient.invalidateQueries({ queryKey: ['allTeamsForLogFilters'] });
    queryClient.invalidateQueries({ queryKey: ['allUsersForLogFilters'] });
    queryClient.invalidateQueries({ queryKey: ['allModels'] });
    
    // Force an immediate refetch of the logs query
    queryClient.refetchQueries({ 
      queryKey: logsQueryKey,
      exact: true,
      type: 'active'
    }).then(() => {
      setIsManualRefreshing(false); // Reset loading state when refresh completes
    });
  };

  return {
    filters,
    filteredLogs: logsQuery.data?.data || [],
    allKeyAliases,
    allTeams: allTeams || [],
    allUsers: allUsers || [],
    allOrganizations: [], 
    allModels: allModels || [],
    handleFilterChange,
    handleFilterReset,
    isLoading: logsQuery.isFetching || isManualRefreshing,
    pagination,
    setCurrentPage, 
    error: logsQuery.error,
    handleRefresh
  };
} 