import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Team } from "@/components/key_team_helpers/key_list";
import { AllModelsDataTable } from "@/components/model_dashboard/all_models_table";
import { columns } from "@/components/molecules/models/columns";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import { InfoCircleOutlined } from "@ant-design/icons";
import { PaginationState, SortingState } from "@tanstack/react-table";
import { Grid, TabPanel } from "@tremor/react";
import { Badge, Select, Skeleton, Space, Typography } from "antd";
import debounce from "lodash/debounce";
import { useEffect, useMemo, useState } from "react";
import { useModelsInfo } from "../../hooks/models/useModels";
import { transformModelData } from "../utils/modelDataTransformer";
type ModelViewMode = "all" | "current_team";
const { Text } = Typography;

interface AllModelsTabProps {
  selectedModelGroup: string | null;
  setSelectedModelGroup: (selectedModelGroup: string) => void;
  availableModelGroups: string[];
  availableModelAccessGroups: string[];
  setSelectedModelId: (id: string) => void;
  setSelectedTeamId: (id: string) => void;
}

const AllModelsTab = ({
  selectedModelGroup,
  setSelectedModelGroup,
  availableModelGroups,
  availableModelAccessGroups,
  setSelectedModelId,
  setSelectedTeamId,
}: AllModelsTabProps) => {
  const { data: modelCostMapData, isLoading: isLoadingModelCostMap } = useModelCostMap();
  const { userId, userRole, premiumUser } = useAuthorized();
  const { data: teams, isLoading: isLoadingTeams } = useTeams();

  const [modelNameSearch, setModelNameSearch] = useState<string>("");
  const [debouncedSearch, setDebouncedSearch] = useState<string>("");
  const [modelViewMode, setModelViewMode] = useState<ModelViewMode>("current_team");
  const [currentTeam, setCurrentTeam] = useState<Team | "personal">("personal");
  const [showFilters, setShowFilters] = useState<boolean>(false);
  const [selectedModelAccessGroupFilter, setSelectedModelAccessGroupFilter] = useState<string | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageSize] = useState<number>(50);
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 50,
  });
  const [sorting, setSorting] = useState<SortingState>([]);

  // Debounce search input
  const debouncedUpdateSearch = useMemo(
    () =>
      debounce((value: string) => {
        setDebouncedSearch(value);
        // Reset to page 1 when search changes
        setCurrentPage(1);
        setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
      }, 200),
    []
  );

  useEffect(() => {
    debouncedUpdateSearch(modelNameSearch);
    return () => {
      debouncedUpdateSearch.cancel();
    };
  }, [modelNameSearch, debouncedUpdateSearch]);

  // Determine teamId to pass to the query - only pass if not "personal"
  const teamIdForQuery = currentTeam === "personal" ? undefined : currentTeam.team_id;

  // Convert sorting state to sortBy and sortOrder for API
  const sortBy = useMemo(() => {
    if (sorting.length === 0) return undefined;
    const sort = sorting[0];
    const columnIdToServerField: Record<string, string> = {
      input_cost: "costs", // Map input_cost column to "costs" for server-side sorting
      model_info_db_model: "status", // Map model_info.db_model column to "status" for server-side sorting
      model_info_created_by: "created_at", // Map model_info.created_by column to "created_at" for server-side sorting
      model_info_updated_at: "updated_at", // Map model_info.updated_at column to "updated_at" for server-side sorting
    };
    return columnIdToServerField[sort.id] || sort.id;
  }, [sorting]);

  const sortOrder = useMemo(() => {
    if (sorting.length === 0) return undefined;
    const sort = sorting[0];
    return sort.desc ? "desc" : "asc";
  }, [sorting]);

  const { data: rawModelData, isLoading: isLoadingModelsInfo } = useModelsInfo(
    currentPage,
    pageSize,
    debouncedSearch || undefined,
    undefined,
    teamIdForQuery,
    sortBy,
    sortOrder
  );
  const isLoading = isLoadingModelsInfo || isLoadingModelCostMap;

  const getProviderFromModel = (model: string) => {
    if (modelCostMapData !== null && modelCostMapData !== undefined) {
      if (typeof modelCostMapData == "object" && model in modelCostMapData) {
        return modelCostMapData[model]["litellm_provider"];
      }
    }
    return "openai";
  };

  const modelData = useMemo(() => {
    if (!rawModelData) return { data: [] };
    return transformModelData(rawModelData, getProviderFromModel);
  }, [rawModelData, modelCostMapData]);

  // Get pagination metadata from the response
  const paginationMeta = useMemo(() => {
    if (!rawModelData) {
      return {
        total_count: 0,
        current_page: 1,
        total_pages: 1,
        size: pageSize,
      };
    }
    return {
      total_count: rawModelData.total_count ?? 0,
      current_page: rawModelData.current_page ?? 1,
      total_pages: rawModelData.total_pages ?? 1,
      size: rawModelData.size ?? pageSize,
    };
  }, [rawModelData, pageSize]);

  const filteredData = useMemo(() => {
    if (!modelData || !modelData.data || modelData.data.length === 0) {
      return [];
    }

    // Server-side search is now handled by the API, so we only filter by other criteria
    return modelData.data.filter((model: any) => {
      const modelNameMatch =
        selectedModelGroup === "all" ||
        model.model_name === selectedModelGroup ||
        !selectedModelGroup ||
        (selectedModelGroup === "wildcard" && model.model_name?.includes("*"));

      const accessGroupMatch =
        selectedModelAccessGroupFilter === "all" ||
        model.model_info["access_groups"]?.includes(selectedModelAccessGroupFilter) ||
        !selectedModelAccessGroupFilter;

      // Team filtering is now handled server-side via teamId query parameter
      // Only apply client-side filtering for model groups and access groups
      return modelNameMatch && accessGroupMatch;
    });
  }, [modelData, selectedModelGroup, selectedModelAccessGroupFilter]);

  useEffect(() => {
    setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
    setCurrentPage(1);
  }, [selectedModelGroup, selectedModelAccessGroupFilter]);

  // Reset pagination when team changes
  useEffect(() => {
    setCurrentPage(1);
    setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
  }, [teamIdForQuery]);

  // Reset pagination when sorting changes
  useEffect(() => {
    setCurrentPage(1);
    setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
  }, [sorting]);

  const resetFilters = () => {
    setModelNameSearch("");
    setSelectedModelGroup("all");
    setSelectedModelAccessGroupFilter(null);
    setCurrentTeam("personal");
    setModelViewMode("current_team");
    setCurrentPage(1);
    setPagination({ pageIndex: 0, pageSize: 50 });
    setSorting([]);
  };

  return (
    <TabPanel>
      <Grid>
        <div className="flex flex-col space-y-4">
          <div className="bg-white rounded-lg shadow">
            {/* Current Team and View Mode Selector - Prominent Section */}
            <div className="border-b px-6 py-4 bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <Text className="text-lg font-semibold text-gray-900">Current Team:</Text>
                  <div className="w-80">
                    {isLoading ? (
                      <Skeleton.Input active block size="large" />
                    ) : (
                      <Select
                        style={{ width: "100%" }}
                        size="large"
                        defaultValue="personal"
                        value={currentTeam === "personal" ? "personal" : currentTeam.team_id}
                        onChange={(value) => {
                          if (value === "personal") {
                            setCurrentTeam("personal");
                            // Reset to page 1 when team changes
                            setCurrentPage(1);
                            setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
                          } else {
                            const team = teams?.find((t) => t.team_id === value);
                            if (team) {
                              setCurrentTeam(team);
                              // Reset to page 1 when team changes
                              setCurrentPage(1);
                              setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
                            }
                          }
                        }}
                        loading={isLoadingTeams}
                        options={[
                          {
                            value: "personal",
                            label: (
                              <Space direction="horizontal" align="center">
                                <Badge color="blue" size="small" />
                                <Text style={{ fontSize: 16 }}>Personal</Text>
                              </Space>
                            ),
                          },
                          ...(teams
                            ?.filter((team) => team.team_id)
                            .map((team) => ({
                              value: team.team_id,
                              label: (
                                <Space direction="horizontal" align="center">
                                  <Badge color="green" size="small" />
                                  <Text ellipsis style={{ fontSize: 16 }}>
                                    {team.team_alias ? team.team_alias : team.team_id}
                                  </Text>
                                </Space>
                              ),
                            })) ?? []),
                        ]}
                      />
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <Text className="text-lg font-semibold text-gray-900">View:</Text>
                  <div className="w-64">
                    {isLoading ? (
                      <Skeleton.Input active block size="large" />
                    ) : (
                      <Select
                        style={{ width: "100%" }}
                        size="large"
                        defaultValue="current_team"
                        value={modelViewMode}
                        onChange={(value) => setModelViewMode(value as "current_team" | "all")}
                        options={[
                          {
                            value: "current_team",
                            label: (
                              <Space direction="horizontal" align="center">
                                <Badge color="purple" size="small" />
                                <Text style={{ fontSize: 16 }}>Current Team Models</Text>
                              </Space>
                            ),
                          },
                          {
                            value: "all",
                            label: (
                              <Space direction="horizontal" align="center">
                                <Badge color="gray" size="small" />
                                <Text style={{ fontSize: 16 }}>All Available Models</Text>
                              </Space>
                            ),
                          },
                        ]}
                      />
                    )}
                  </div>
                </div>
              </div>

              {modelViewMode === "current_team" && (
                <div className="flex items-start gap-2 mt-3">
                  <InfoCircleOutlined className="text-gray-400 mt-0.5 flex-shrink-0 text-xs" />
                  <div className="text-xs text-gray-500">
                    {currentTeam === "personal" ? (
                      <span>
                        To access these models: Create a Virtual Key without selecting a team on the{" "}
                        <a
                          href="/public?login=success&page=api-keys"
                          className="text-gray-600 hover:text-gray-800 underline"
                        >
                          Virtual Keys page
                        </a>
                      </span>
                    ) : (
                      <span>
                        To access these models: Create a Virtual Key and select Team as &quot;
                        {typeof currentTeam !== "string" ? currentTeam.team_alias || currentTeam.team_id : ""}&quot; on
                        the{" "}
                        <a
                          href="/public?login=success&page=api-keys"
                          className="text-gray-600 hover:text-gray-800 underline"
                        >
                          Virtual Keys page
                        </a>
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Search and Filter Controls */}
            <div className="border-b px-6 py-4">
              <div className="flex flex-col space-y-4">
                {/* Search and Filter Controls */}
                <div className="flex flex-wrap items-center gap-3">
                  {/* Model Name Search */}
                  <div className="relative w-64">
                    <input
                      type="text"
                      placeholder="Search model names..."
                      className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      value={modelNameSearch}
                      onChange={(e) => setModelNameSearch(e.target.value)}
                    />
                    <svg
                      className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                      />
                    </svg>
                  </div>

                  {/* Filter Button */}
                  <button
                    className={`px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 ${showFilters ? "bg-gray-100" : ""}`}
                    onClick={() => setShowFilters(!showFilters)}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                      />
                    </svg>
                    Filters
                  </button>

                  {/* Reset Filters Button */}
                  <button
                    className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                    onClick={resetFilters}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                    Reset Filters
                  </button>
                </div>

                {/* Additional Filters */}
                {showFilters && (
                  <div className="flex flex-wrap items-center gap-3 mt-3">
                    {/* Model Name Filter */}
                    <div className="w-64">
                      <Select
                        className="w-full"
                        value={selectedModelGroup ?? "all"}
                        onChange={(value) => setSelectedModelGroup(value === "all" ? "all" : value)}
                        placeholder="Filter by Public Model Name"
                        showSearch
                        options={[
                          { value: "all", label: "All Models" },
                          { value: "wildcard", label: "Wildcard Models (*)" },
                          ...availableModelGroups.map((group, idx) => ({
                            value: group,
                            label: group,
                          })),
                        ]}
                      />
                    </div>

                    {/* Model Access Group Filter */}
                    <div className="w-64">
                      <Select
                        className="w-full"
                        value={selectedModelAccessGroupFilter ?? "all"}
                        onChange={(value) => setSelectedModelAccessGroupFilter(value === "all" ? null : value)}
                        placeholder="Filter by Model Access Group"
                        showSearch
                        options={[
                          { value: "all", label: "All Model Access Groups" },
                          ...availableModelAccessGroups.map((accessGroup, idx) => ({
                            value: accessGroup,
                            label: accessGroup,
                          })),
                        ]}
                      />
                    </div>
                  </div>
                )}

                {/* Results Count and Pagination Controls */}
                <div className="flex justify-between items-center">
                  {isLoading ? (
                    <Skeleton.Input active style={{ width: 184, height: 20 }} />
                  ) : (
                    <span className="text-sm text-gray-700">
                      {paginationMeta.total_count > 0
                        ? `Showing ${((currentPage - 1) * pageSize) + 1} - ${Math.min(currentPage * pageSize, paginationMeta.total_count)} of ${paginationMeta.total_count} results`
                        : "Showing 0 results"}
                    </span>
                  )}

                  <div className="flex items-center space-x-2">
                    {isLoading ? (
                      <Skeleton.Button active style={{ width: 84, height: 30 }} />
                    ) : (
                      <button
                        onClick={() => {
                          const newPage = currentPage - 1;
                          setCurrentPage(newPage);
                          setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
                        }}
                        disabled={currentPage === 1}
                        className={`px-3 py-1 text-sm border rounded-md ${currentPage === 1
                          ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                          : "hover:bg-gray-50"
                          }`}
                      >
                        Previous
                      </button>
                    )}

                    {isLoading ? (
                      <Skeleton.Button active style={{ width: 56, height: 30 }} />
                    ) : (
                      <button
                        onClick={() => {
                          const newPage = currentPage + 1;
                          setCurrentPage(newPage);
                          setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
                        }}
                        disabled={currentPage >= paginationMeta.total_pages}
                        className={`px-3 py-1 text-sm border rounded-md ${currentPage >= paginationMeta.total_pages
                          ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                          : "hover:bg-gray-50"
                          }`}
                      >
                        Next
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <AllModelsDataTable
              columns={columns(
                userRole,
                userId,
                premiumUser,
                setSelectedModelId,
                setSelectedTeamId,
                getDisplayModelName,
                () => { },
                () => { },
                expandedRows,
                setExpandedRows,
              )}
              data={filteredData}
              isLoading={isLoadingModelsInfo}
              sorting={sorting}
              onSortingChange={setSorting}
              pagination={pagination}
              onPaginationChange={setPagination}
              enablePagination={true}
            />
          </div>
        </div>
      </Grid>
    </TabPanel>
  );
};

export default AllModelsTab;
