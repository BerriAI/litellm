import { DataTable } from "./table";
import moment from "moment";
import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { getTimeRangeDisplay } from "./logs_utils";
import { useQuery } from "@tanstack/react-query";
import { uiAuditLogsCall, keyListCall } from "../networking";
import { AuditLogEntry, auditLogColumns } from "./columns";
import { Text } from "@tremor/react";
import FilterComponent, { FilterOption } from "../common_components/filter";
import { Team } from "../key_team_helpers/key_list";
import { fetchAllKeyAliases } from "../key_team_helpers/filter_helpers";

interface AuditLogsProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  isActive: boolean;
  premiumUser: boolean;
  allTeams: Team[];
}

export default function AuditLogs({
  userID,
  userRole,
  token,
  accessToken,
  isActive,
  premiumUser,
  allTeams,
}: AuditLogsProps) {
  const [startTime, setStartTime] = useState<string>(
    moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm")
  );
  const [endTime, setEndTime] = useState<string>(
    moment().format("YYYY-MM-DDTHH:mm")
  );

  const quickSelectRef = useRef<HTMLDivElement>(null);
  const [quickSelectOpen, setQuickSelectOpen] = useState(false);
  const [isCustomDate, setIsCustomDate] = useState(false);
  const [clientCurrentPage, setClientCurrentPage] = useState(1);
  const [pageSize] = useState(50);
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [selectedTeamId, setSelectedTeamId] = useState("");
  const [selectedKeyHash, setSelectedKeyHash] = useState("");

  const allLogsQuery = useQuery<AuditLogEntry[]>({
    queryKey: [
      "all_audit_logs",
      accessToken,
      token,
      userRole,
      userID,
      startTime,
      endTime,
      isCustomDate,
    ],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        return [];
      }

      const formattedStartTimeStr = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTimeStr = isCustomDate
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      let accumulatedLogs: AuditLogEntry[] = [];
      let currentPageToFetch = 1;
      let totalPagesFromBackend = 1;
      const backendPageSize = 50;

      do {
        const response = await uiAuditLogsCall(
          accessToken,
          formattedStartTimeStr,
          formattedEndTimeStr,
          currentPageToFetch,
          backendPageSize
        );
        accumulatedLogs = accumulatedLogs.concat(response.audit_logs);
        totalPagesFromBackend = response.total_pages;
        currentPageToFetch++;
      } while (currentPageToFetch <= totalPagesFromBackend);

      return accumulatedLogs;
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && isActive,
  });

  const handleRefresh = () => {
    allLogsQuery.refetch();
  };
  
  const handleFilterChange = (newFilters: Record<string, string>) => {
    setFilters(newFilters);
  };

  const handleFilterReset = () => {
    setFilters({});
    setSelectedTeamId("");
    setSelectedKeyHash("");
    setClientCurrentPage(1);
  };

  const fetchKeyHashForAlias = useCallback(async (keyAlias: string) => {
    if (!accessToken) return;
    
    try {
      const response = await keyListCall(
        accessToken,
        null,
        null,
        keyAlias,
        null,
        null,
        1,
        10
      );

      const selectedKey = response.keys.find(
        (key: any) => key.key_alias === keyAlias
      );

      if (selectedKey) {
        setSelectedKeyHash(selectedKey.token);
      } else {
        setSelectedKeyHash("");
      }
    } catch (error) {
      console.error("Error fetching key hash for alias:", error);
      setSelectedKeyHash("");
    }
  }, [accessToken]);

  useEffect(() => {
    if(!accessToken) return;

    let teamIdChanged = false;
    let keyHashChanged = false;

    if (filters['Team ID']) {
      if (selectedTeamId !== filters['Team ID']) {
        setSelectedTeamId(filters['Team ID']);
        teamIdChanged = true;
      }
    } else {
      if (selectedTeamId !== "") {
        setSelectedTeamId("");
        teamIdChanged = true;
      }
    }
    
    if (filters['Key Hash']) {
      if (selectedKeyHash !== filters['Key Hash']) {
        setSelectedKeyHash(filters['Key Hash']);
        keyHashChanged = true;
      }
    } else if (filters['Key Alias']) {
      fetchKeyHashForAlias(filters['Key Alias']);
    } else {
      if (selectedKeyHash !== "") {
        setSelectedKeyHash("");
        keyHashChanged = true;
      }
    }

    if (teamIdChanged || keyHashChanged) {
        setClientCurrentPage(1);
    }
  }, [filters, accessToken, fetchKeyHashForAlias, selectedTeamId, selectedKeyHash]);

  useEffect(() => {
    setClientCurrentPage(1);
  }, [selectedTeamId, selectedKeyHash, startTime, endTime]);

  const completeFilteredLogs = useMemo(() => {
    if (!allLogsQuery.data) return [];
    return allLogsQuery.data.filter(log => {
      let matchesTeam = true;
      let matchesKey = true;

      if (selectedTeamId) {
        const beforeTeamId = typeof log.before_value === 'string' ? JSON.parse(log.before_value)?.team_id : log.before_value?.team_id;
        const updatedTeamId = typeof log.updated_values === 'string' ? JSON.parse(log.updated_values)?.team_id : log.updated_values?.team_id;
        matchesTeam = beforeTeamId === selectedTeamId || updatedTeamId === selectedTeamId;
      }

      if (selectedKeyHash) {
        try {
          const beforeBody = typeof log.before_value === 'string' ? JSON.parse(log.before_value) : log.before_value;
          const updatedBody = typeof log.updated_values === 'string' ? JSON.parse(log.updated_values) : log.updated_values;
          
          const beforeKey = beforeBody?.token;
          const updatedKey = updatedBody?.token;

          matchesKey = (typeof beforeKey === 'string' && beforeKey.includes(selectedKeyHash)) ||
                      (typeof updatedKey === 'string' && updatedKey.includes(selectedKeyHash));
        } catch (e) {
          matchesKey = false;
        }
      }

      return matchesTeam && matchesKey;
    });
  }, [allLogsQuery.data, selectedTeamId, selectedKeyHash]);

  const totalFilteredItems = completeFilteredLogs.length;
  const totalFilteredPages = Math.ceil(totalFilteredItems / pageSize) || 1;

  const paginatedViewOfFilteredLogs = useMemo(() => {
    const start = (clientCurrentPage - 1) * pageSize;
    const end = start + pageSize;
    return completeFilteredLogs.slice(start, end);
  }, [completeFilteredLogs, clientCurrentPage, pageSize]);

  if (!premiumUser) {
    return (
      <div>
        <Text>This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key <a href="https://litellm.ai/pricing" target="_blank" rel="noopener noreferrer">here</a>.</Text>
      </div>
    );
  }

  const auditLogFilterOptions: FilterOption[] = [
    {
      name: 'Team ID',
      label: 'Team ID',
      isSearchable: true,
      searchFn: async (searchText: string) => {
        if (!allTeams || allTeams.length === 0) return [];
        const filtered = allTeams.filter((team: Team) =>{
          return team.team_id.toLowerCase().includes(searchText.toLowerCase()) ||
          (team.team_alias && team.team_alias.toLowerCase().includes(searchText.toLowerCase()))
        });
        return filtered.map((team: Team) => ({
          label: `${team.team_alias || team.team_id} (${team.team_id})`,
          value: team.team_id
        }));
      }
    },
    {
      name: 'Key Alias',
      label: 'Key Alias',
      isSearchable: true,
      searchFn: async (searchText: string) => {
        if (!accessToken) return [];
        const keyAliases = await fetchAllKeyAliases(accessToken);
        const filtered = keyAliases.filter(alias => 
          alias.toLowerCase().includes(searchText.toLowerCase())
        );
        return filtered.map(alias => ({
          label: alias,
          value: alias
        }));
      }
    },
    {
      name: 'Key Hash',
      label: 'Key Hash',
      isSearchable: false,
    },
  ];

  const currentDisplayItemsStart = totalFilteredItems > 0 ? (clientCurrentPage - 1) * pageSize + 1 : 0;
  const currentDisplayItemsEnd = Math.min(clientCurrentPage * pageSize, totalFilteredItems);

  return (
    <>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">
          Audit Logs
        </h1>
      </div>
      <FilterComponent options={auditLogFilterOptions} onApplyFilters={handleFilterChange} onResetFilters={handleFilterReset} />
      <div className="bg-white rounded-lg shadow">
        <div className="border-b px-6 py-4">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0">
            <div className="flex flex-wrap items-center gap-3">
              

              <div className="flex items-center gap-2">
                <div className="relative" ref={quickSelectRef}>
                </div>
                
                <button
                  onClick={handleRefresh}
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                  title="Refresh data"
                >
                  <svg
                    className={`w-4 h-4 ${allLogsQuery.isFetching ? 'animate-spin' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                  <span>Refresh</span>
                </button>
              </div>

              {isCustomDate && (
                <div className="flex items-center gap-2">
                  <div>
                    <input
                      type="datetime-local"
                      value={startTime}
                      onChange={(e) => {
                        setStartTime(e.target.value);
                      }}
                      className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <span className="text-gray-500">to</span>
                  <div>
                    <input
                      type="datetime-local"
                      value={endTime}
                      onChange={(e) => {
                        setEndTime(e.target.value);
                      }}
                      className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">
                Showing{" "}
                {allLogsQuery.isLoading
                  ? "..."
                  : currentDisplayItemsStart}{" "}
                -{" "}
                {allLogsQuery.isLoading
                  ? "..."
                  : currentDisplayItemsEnd}{" "}
                of{" "}
                {allLogsQuery.isLoading
                  ? "..."
                  : totalFilteredItems}{" "}
                results
              </span>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-700">
                  Page {allLogsQuery.isLoading ? "..." : clientCurrentPage} of{" "}
                  {allLogsQuery.isLoading
                    ? "..."
                    : totalFilteredPages}
                </span>
                <button
                  onClick={() =>
                    setClientCurrentPage((p) => Math.max(1, p - 1))
                  }
                  disabled={allLogsQuery.isLoading || clientCurrentPage === 1}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() =>
                    setClientCurrentPage((p) =>
                      Math.min(
                        totalFilteredPages,
                        p + 1,
                      ),
                    )
                  }
                  disabled={
                    allLogsQuery.isLoading ||
                    clientCurrentPage === totalFilteredPages
                  }
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>
        <DataTable
          columns={auditLogColumns}
          data={paginatedViewOfFilteredLogs}
          renderSubComponent={() => {return <></>}}
          getRowCanExpand={() => false}
        />
      </div>
    </>
  );
}
