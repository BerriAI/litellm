import { DataTable } from "./table";
import moment from "moment";
import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { uiAuditLogsCall, keyListCall } from "../networking";
import { AuditLogEntry, auditLogColumns } from "./columns";
import { Text } from "@tremor/react";
import { Team } from "../key_team_helpers/key_list";
import { formatNumberWithCommas } from "@/utils/dataUtils";

interface AuditLogsProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  isActive: boolean;
  premiumUser: boolean;
  allTeams: Team[];
}

const asset_logos_folder = "../ui/assets/";
export const auditLogsPreviewImg = `${asset_logos_folder}audit-logs-preview.png`;

export default function AuditLogs({
  userID,
  userRole,
  token,
  accessToken,
  isActive,
  premiumUser,
  allTeams,
}: AuditLogsProps) {
  const [startTime, setStartTime] = useState<string>(moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm"));

  const actionFilterRef = useRef<HTMLDivElement>(null);
  const tableFilterRef = useRef<HTMLDivElement>(null);
  const [clientCurrentPage, setClientCurrentPage] = useState(1);
  const [pageSize] = useState(50);
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [selectedTeamId, setSelectedTeamId] = useState("");
  const [selectedKeyHash, setSelectedKeyHash] = useState("");
  const [objectIdSearch, setObjectIdSearch] = useState("");
  const [selectedActionFilter, setSelectedActionFilter] = useState("all");
  const [selectedTableFilter, setSelectedTableFilter] = useState("all");
  const [actionFilterOpen, setActionFilterOpen] = useState(false);
  const [tableFilterOpen, setTableFilterOpen] = useState(false);

  const allLogsQuery = useQuery<AuditLogEntry[]>({
    queryKey: ["all_audit_logs", accessToken, token, userRole, userID, startTime],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        return [];
      }

      const formattedStartTimeStr = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTimeStr = moment().utc().format("YYYY-MM-DD HH:mm:ss");

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
          backendPageSize,
        );
        accumulatedLogs = accumulatedLogs.concat(response.audit_logs);
        totalPagesFromBackend = response.total_pages;
        currentPageToFetch++;
      } while (currentPageToFetch <= totalPagesFromBackend);

      return accumulatedLogs;
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && isActive,
    refetchInterval: 5000,
    refetchIntervalInBackground: true,
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
    setObjectIdSearch("");
    setSelectedActionFilter("all");
    setSelectedTableFilter("all");
    setClientCurrentPage(1);
  };

  const fetchKeyHashForAlias = useCallback(
    async (keyAlias: string) => {
      if (!accessToken) return;

      try {
        const response = await keyListCall(accessToken, null, null, keyAlias, null, null, 1, 10);

        const selectedKey = response.keys.find((key: any) => key.key_alias === keyAlias);

        if (selectedKey) {
          setSelectedKeyHash(selectedKey.token);
        } else {
          setSelectedKeyHash("");
        }
      } catch (error) {
        console.error("Error fetching key hash for alias:", error);
        setSelectedKeyHash("");
      }
    },
    [accessToken],
  );

  useEffect(() => {
    if (!accessToken) return;

    let teamIdChanged = false;
    let keyHashChanged = false;

    if (filters["Team ID"]) {
      if (selectedTeamId !== filters["Team ID"]) {
        setSelectedTeamId(filters["Team ID"]);
        teamIdChanged = true;
      }
    } else {
      if (selectedTeamId !== "") {
        setSelectedTeamId("");
        teamIdChanged = true;
      }
    }

    if (filters["Key Hash"]) {
      if (selectedKeyHash !== filters["Key Hash"]) {
        setSelectedKeyHash(filters["Key Hash"]);
        keyHashChanged = true;
      }
    } else if (filters["Key Alias"]) {
      fetchKeyHashForAlias(filters["Key Alias"]);
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
  }, [selectedTeamId, selectedKeyHash, startTime, objectIdSearch, selectedActionFilter, selectedTableFilter]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (actionFilterRef.current && !actionFilterRef.current.contains(event.target as Node)) {
        setActionFilterOpen(false);
      }
      if (tableFilterRef.current && !tableFilterRef.current.contains(event.target as Node)) {
        setTableFilterOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const completeFilteredLogs = useMemo(() => {
    if (!allLogsQuery.data) return [];
    return allLogsQuery.data.filter((log) => {
      let matchesTeam = true;
      let matchesKey = true;
      let matchesObjectId = true;
      let matchesAction = true;
      let matchesTable = true;

      if (selectedTeamId) {
        const beforeTeamId =
          typeof log.before_value === "string" ? JSON.parse(log.before_value)?.team_id : log.before_value?.team_id;
        const updatedTeamId =
          typeof log.updated_values === "string"
            ? JSON.parse(log.updated_values)?.team_id
            : log.updated_values?.team_id;
        matchesTeam = beforeTeamId === selectedTeamId || updatedTeamId === selectedTeamId;
      }

      if (selectedKeyHash) {
        try {
          const beforeBody = typeof log.before_value === "string" ? JSON.parse(log.before_value) : log.before_value;
          const updatedBody =
            typeof log.updated_values === "string" ? JSON.parse(log.updated_values) : log.updated_values;

          const beforeKey = beforeBody?.token;
          const updatedKey = updatedBody?.token;

          matchesKey =
            (typeof beforeKey === "string" && beforeKey.includes(selectedKeyHash)) ||
            (typeof updatedKey === "string" && updatedKey.includes(selectedKeyHash));
        } catch (e) {
          matchesKey = false;
        }
      }

      if (objectIdSearch) {
        matchesObjectId = log.object_id?.toLowerCase().includes(objectIdSearch.toLowerCase());
      }

      if (selectedActionFilter !== "all") {
        matchesAction = log.action?.toLowerCase() === selectedActionFilter.toLowerCase();
      }

      if (selectedTableFilter !== "all") {
        let tableMatchName = "";
        switch (selectedTableFilter) {
          case "keys":
            tableMatchName = "litellm_verificationtoken";
            break;
          case "teams":
            tableMatchName = "litellm_teamtable";
            break;
          case "users":
            tableMatchName = "litellm_usertable";
            break;
          // Add other direct table names if needed, or rely on a more generic match
          default:
            tableMatchName = selectedTableFilter; // Should not happen with current UI options
        }
        matchesTable = log.table_name?.toLowerCase() === tableMatchName;
      }

      return matchesTeam && matchesKey && matchesObjectId && matchesAction && matchesTable;
    });
  }, [allLogsQuery.data, selectedTeamId, selectedKeyHash, objectIdSearch, selectedActionFilter, selectedTableFilter]);

  const totalFilteredItems = completeFilteredLogs.length;
  const totalFilteredPages = Math.ceil(totalFilteredItems / pageSize) || 1;

  const paginatedViewOfFilteredLogs = useMemo(() => {
    const start = (clientCurrentPage - 1) * pageSize;
    const end = start + pageSize;
    return completeFilteredLogs.slice(start, end);
  }, [completeFilteredLogs, clientCurrentPage, pageSize]);

  // Check if audit logs are empty (not loading and no data)
  const showAuditLogsInfo = !allLogsQuery.data || allLogsQuery.data.length === 0;

  // Custom AuditLogsInfoMessage component
  const AuditLogsInfoMessage = ({ show }: { show: boolean }) => {
    if (!show) return null;

    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start mb-6">
        <div className="text-blue-500 mr-3 flex-shrink-0 mt-0.5">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="16" x2="12" y2="12"></line>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
          </svg>
        </div>
        <div>
          <h4 className="text-sm font-medium text-blue-800">Audit Logs Not Available</h4>
          <p className="text-sm text-blue-700 mt-1">
            To enable audit logging, add the following configuration to your LiteLLM proxy configuration file:
          </p>
          <pre className="mt-2 bg-white p-3 rounded border border-blue-200 text-xs font-mono overflow-auto">
            {`litellm_settings:
  store_audit_logs: true`}
          </pre>
          <p className="text-xs text-blue-700 mt-2">
            Note: This will only affect new requests after the configuration change and proxy restart.
          </p>
        </div>
      </div>
    );
  };

  const renderSubComponent = useCallback(({ row }: { row: any }) => {
    const AuditLogRowExpansionPanel = ({ rowData }: { rowData: AuditLogEntry }) => {
      const { before_value, updated_values, table_name, action } = rowData;

      const renderValue = (value: Record<string, any>, isKeyTable: boolean) => {
        if (!value || Object.keys(value).length === 0) return <Text>N/A</Text>;

        if (isKeyTable) {
          const changedKeys = Object.keys(value);
          const knownKeyFields = ["token", "spend", "max_budget"];

          const onlyKnownFieldsChanged = changedKeys.every((key) => knownKeyFields.includes(key));

          if (onlyKnownFieldsChanged && changedKeys.length > 0) {
            return (
              <div>
                {changedKeys.includes("token") && (
                  <p>
                    <strong>Token:</strong> {value.token || "N/A"}
                  </p>
                )}
                {changedKeys.includes("spend") && (
                  <p>
                    <strong>Spend:</strong>{" "}
                    {value.spend !== undefined ? `$${formatNumberWithCommas(value.spend, 6)}` : "N/A"}
                  </p>
                )}
                {changedKeys.includes("max_budget") && (
                  <p>
                    <strong>Max Budget:</strong>{" "}
                    {value.max_budget !== undefined ? `$${formatNumberWithCommas(value.max_budget, 6)}` : "N/A"}
                  </p>
                )}
              </div>
            );
          } else {
            if (
              value["No differing fields detected in 'before' state"] ||
              value["No differing fields detected in 'updated' state"] ||
              value["No fields changed"]
            ) {
              return <Text>{value[Object.keys(value)[0]]}</Text>; // Display the N/A message string
            }
            return (
              <pre className="p-2 bg-gray-50 border rounded text-xs overflow-auto max-h-60">
                {JSON.stringify(value, null, 2)}
              </pre>
            );
          }
        }

        return (
          <pre className="p-2 bg-gray-50 border rounded text-xs overflow-auto max-h-60">
            {JSON.stringify(value, null, 2)}
          </pre>
        );
      };

      let displayBeforeValue = before_value;
      let displayUpdatedValue = updated_values;

      if ((action === "updated" || action === "rotated") && before_value && updated_values) {
        if (
          table_name === "LiteLLM_TeamTable" ||
          table_name === "LiteLLM_UserTable" ||
          table_name === "LiteLLM_VerificationToken"
        ) {
          const changedBefore: Record<string, any> = {};
          const changedUpdated: Record<string, any> = {};
          const allKeys = new Set([...Object.keys(before_value), ...Object.keys(updated_values)]);

          allKeys.forEach((key) => {
            const beforeValStr = JSON.stringify(before_value[key]);
            const updatedValStr = JSON.stringify(updated_values[key]);
            if (beforeValStr !== updatedValStr) {
              if (before_value.hasOwnProperty(key)) {
                changedBefore[key] = before_value[key];
              }
              if (updated_values.hasOwnProperty(key)) {
                changedUpdated[key] = updated_values[key];
              }
            }
          });

          Object.keys(before_value).forEach((key) => {
            if (!updated_values.hasOwnProperty(key) && !changedBefore.hasOwnProperty(key)) {
              changedBefore[key] = before_value[key];
              changedUpdated[key] = undefined;
            }
          });

          Object.keys(updated_values).forEach((key) => {
            if (!before_value.hasOwnProperty(key) && !changedUpdated.hasOwnProperty(key)) {
              changedUpdated[key] = updated_values[key];
              changedBefore[key] = undefined;
            }
          });

          displayBeforeValue =
            Object.keys(changedBefore).length > 0
              ? changedBefore
              : { "No differing fields detected in 'before' state": "N/A" };
          displayUpdatedValue =
            Object.keys(changedUpdated).length > 0
              ? changedUpdated
              : { "No differing fields detected in 'updated' state": "N/A" };

          if (Object.keys(changedBefore).length === 0 && Object.keys(changedUpdated).length === 0) {
            displayBeforeValue = { "No fields changed": "N/A" };
            displayUpdatedValue = { "No fields changed": "N/A" };
          }
        }
      }

      return (
        <div className="-mx-4 p-4 bg-slate-100 border-y border-slate-300 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold mb-2 text-sm text-slate-700">Before Value:</h4>
            {renderValue(displayBeforeValue, table_name === "LiteLLM_VerificationToken")}
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-sm text-slate-700">Updated Value:</h4>
            {renderValue(displayUpdatedValue, table_name === "LiteLLM_VerificationToken")}
          </div>
        </div>
      );
    };

    return <AuditLogRowExpansionPanel rowData={row.original as AuditLogEntry} />;
  }, []);

  if (!premiumUser) {
    return (
      <div style={{ textAlign: "center", marginTop: "20px" }}>
        <h1 style={{ display: "block", marginBottom: "10px" }}>✨ Enterprise Feature.</h1>
        <Text style={{ display: "block", marginBottom: "10px" }}>
          This is a LiteLLM Enterprise feature, and requires a valid key to use.
        </Text>
        <Text style={{ display: "block", marginBottom: "20px", fontStyle: "italic" }}>
          Here&apos;s a preview of what Audit Logs offer:
        </Text>
        <img
          src={auditLogsPreviewImg}
          alt="Audit Logs Preview"
          style={{
            maxWidth: "100%",
            maxHeight: "700px",
            borderRadius: "8px",
            boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
            margin: "0 auto",
          }}
          onError={(e) => {
            console.error("Failed to load audit logs preview image");
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>
    );
  }

  const currentDisplayItemsStart = totalFilteredItems > 0 ? (clientCurrentPage - 1) * pageSize + 1 : 0;
  const currentDisplayItemsEnd = Math.min(clientCurrentPage * pageSize, totalFilteredItems);

  return (
    <>
      <div className="flex items-center justify-between mb-4"></div>
      {/* <FilterComponent options={auditLogFilterOptions} onApplyFilters={handleFilterChange} onResetFilters={handleFilterReset} /> */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b px-6 py-4">
          <h1 className="text-xl font-semibold py-4">Audit Logs</h1>

          {/* Show Audit Logs Info Message when no data */}
          <AuditLogsInfoMessage show={showAuditLogsInfo} />

          <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center">
                  <input
                    type="text"
                    placeholder="Search by Object ID..."
                    value={objectIdSearch}
                    onChange={(e) => setObjectIdSearch(e.target.value)}
                    className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <button
                  onClick={handleRefresh}
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                  title="Refresh data"
                >
                  <svg
                    className={`w-4 h-4 ${allLogsQuery.isFetching ? "animate-spin" : ""}`}
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
            </div>

            <div className="flex items-center space-x-4">
              {/* Custom Action Filter Dropdown */}
              <div className="relative" ref={actionFilterRef}>
                <label htmlFor="actionFilterDisplay" className="mr-2 text-sm font-medium text-gray-700 sr-only">
                  Action:
                </label>
                <button
                  id="actionFilterDisplay"
                  onClick={() => setActionFilterOpen(!actionFilterOpen)}
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 bg-white w-40 text-left justify-between"
                >
                  <span>
                    {selectedActionFilter === "all" && "All Actions"}
                    {selectedActionFilter === "created" && "Created"}
                    {selectedActionFilter === "updated" && "Updated"}
                    {selectedActionFilter === "deleted" && "Deleted"}
                    {selectedActionFilter === "rotated" && "Rotated"}
                  </span>
                  <svg
                    className="w-4 h-4 text-gray-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </button>
                {actionFilterOpen && (
                  <div className="absolute left-0 mt-2 w-40 bg-white rounded-lg shadow-lg border p-1 z-50">
                    <div className="space-y-1">
                      {[
                        { label: "All Actions", value: "all" },
                        { label: "Created", value: "created" },
                        { label: "Updated", value: "updated" },
                        { label: "Deleted", value: "deleted" },
                        { label: "Rotated", value: "rotated" },
                      ].map((option) => (
                        <button
                          key={option.value}
                          className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md ${
                            selectedActionFilter === option.value
                              ? "bg-blue-50 text-blue-600 font-medium"
                              : "font-normal"
                          }`}
                          onClick={() => {
                            setSelectedActionFilter(option.value);
                            setActionFilterOpen(false);
                          }}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Custom Table Filter Dropdown */}
              <div className="relative" ref={tableFilterRef}>
                <label htmlFor="tableFilterDisplay" className="mr-2 text-sm font-medium text-gray-700 sr-only">
                  Table:
                </label>
                <button
                  id="tableFilterDisplay"
                  onClick={() => setTableFilterOpen(!tableFilterOpen)}
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 bg-white w-40 text-left justify-between"
                >
                  <span>
                    {selectedTableFilter === "all" && "All Tables"}
                    {selectedTableFilter === "keys" && "Keys"}
                    {selectedTableFilter === "teams" && "Teams"}
                    {selectedTableFilter === "users" && "Users"}
                  </span>
                  <svg
                    className="w-4 h-4 text-gray-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </button>
                {tableFilterOpen && (
                  <div className="absolute left-0 mt-2 w-40 bg-white rounded-lg shadow-lg border p-1 z-50">
                    <div className="space-y-1">
                      {[
                        { label: "All Tables", value: "all" },
                        { label: "Keys", value: "keys" },
                        { label: "Teams", value: "teams" },
                        { label: "Users", value: "users" },
                      ].map((option) => (
                        <button
                          key={option.value}
                          className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md ${
                            selectedTableFilter === option.value
                              ? "bg-blue-50 text-blue-600 font-medium"
                              : "font-normal"
                          }`}
                          onClick={() => {
                            setSelectedTableFilter(option.value);
                            setTableFilterOpen(false);
                          }}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <span className="text-sm text-gray-700">
                Showing {allLogsQuery.isLoading ? "..." : currentDisplayItemsStart} -{" "}
                {allLogsQuery.isLoading ? "..." : currentDisplayItemsEnd} of{" "}
                {allLogsQuery.isLoading ? "..." : totalFilteredItems} results
              </span>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-700">
                  Page {allLogsQuery.isLoading ? "..." : clientCurrentPage} of{" "}
                  {allLogsQuery.isLoading ? "..." : totalFilteredPages}
                </span>
                <button
                  onClick={() => setClientCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={allLogsQuery.isLoading || clientCurrentPage === 1}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() => setClientCurrentPage((p) => Math.min(totalFilteredPages, p + 1))}
                  disabled={allLogsQuery.isLoading || clientCurrentPage === totalFilteredPages}
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
          renderSubComponent={renderSubComponent}
          getRowCanExpand={() => true}
        />
      </div>
    </>
  );
}
