import { Grid, Select, SelectItem, TabPanel, Text } from "@tremor/react";
import { InfoCircleOutlined } from "@ant-design/icons";
import { ModelDataTable } from "@/components/model_dashboard/table";
import { columns } from "@/components/molecules/models/columns";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import React, { useEffect, useMemo, useRef, useState } from "react";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Table as TableInstance, PaginationState } from "@tanstack/react-table";

type ModelViewMode = "all" | "current_team";

interface AllModelsTabProps {
  selectedModelGroup: string | null;
  setSelectedModelGroup: (selectedModelGroup: string) => void;
  availableModelGroups: string[];
  availableModelAccessGroups: string[];
  setSelectedModelId: (id: string) => void;
  setSelectedTeamId: (id: string) => void;
  setEditModel: (edit: boolean) => void;
  modelData: any;
}

const AllModelsTab = ({
  selectedModelGroup,
  setSelectedModelGroup,
  availableModelGroups,
  availableModelAccessGroups,
  setSelectedModelId,
  setSelectedTeamId,
  setEditModel,
  modelData,
}: AllModelsTabProps) => {
  const { userId, userRole, premiumUser } = useAuthorized();
  const { teams } = useTeams();

  const [modelNameSearch, setModelNameSearch] = useState<string>("");
  const [modelViewMode, setModelViewMode] = useState<ModelViewMode>("current_team");
  const [currentTeam, setCurrentTeam] = useState<string>("personal"); // 'personal' or team_id
  const [showFilters, setShowFilters] = useState<boolean>(false);
  const [selectedModelAccessGroupFilter, setSelectedModelAccessGroupFilter] = useState<string | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 50,
  });
  const tableRef = useRef<TableInstance<any>>(null);

  const filteredData = useMemo(() => {
    if (!modelData || !modelData.data || modelData.data.length === 0) {
      return [];
    }

    return modelData.data.filter((model: any) => {
      const searchMatch =
        modelNameSearch === "" || model.model_name.toLowerCase().includes(modelNameSearch.toLowerCase());

      const modelNameMatch =
        selectedModelGroup === "all" ||
        model.model_name === selectedModelGroup ||
        !selectedModelGroup ||
        (selectedModelGroup === "wildcard" && model.model_name?.includes("*"));

      const accessGroupMatch =
        selectedModelAccessGroupFilter === "all" ||
        model.model_info["access_groups"]?.includes(selectedModelAccessGroupFilter) ||
        !selectedModelAccessGroupFilter;

      let teamAccessMatch = true;
      if (modelViewMode === "current_team") {
        if (currentTeam === "personal") {
          teamAccessMatch = model.model_info?.direct_access === true;
        } else {
          teamAccessMatch = model.model_info?.access_via_team_ids?.includes(currentTeam) === true;
        }
      }

      return searchMatch && modelNameMatch && accessGroupMatch && teamAccessMatch;
    });
  }, [modelData, modelNameSearch, selectedModelGroup, selectedModelAccessGroupFilter, currentTeam, modelViewMode]);

  const paginatedData = useMemo(() => {
    const startIndex = pagination.pageIndex * pagination.pageSize;
    const endIndex = startIndex + pagination.pageSize;
    return filteredData.slice(startIndex, endIndex);
  }, [filteredData, pagination.pageIndex, pagination.pageSize]);

  useEffect(() => {
    setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
  }, [modelNameSearch, selectedModelGroup, selectedModelAccessGroupFilter, currentTeam, modelViewMode]);

  const resetFilters = () => {
    setModelNameSearch("");
    setSelectedModelGroup("all");
    setSelectedModelAccessGroupFilter(null);
    setCurrentTeam("personal");
    setModelViewMode("current_team");
    setPagination({ pageIndex: 0, pageSize: 50 });
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
                  <Select
                    className="w-80"
                    defaultValue="personal"
                    value={currentTeam}
                    onValueChange={(value) => setCurrentTeam(value)}
                  >
                    <SelectItem value="personal">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <span className="font-medium">Personal</span>
                      </div>
                    </SelectItem>
                    {teams
                      ?.filter((team) => team.team_id)
                      .map((team) => (
                        <SelectItem key={team.team_id} value={team.team_id}>
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="font-medium">
                              {team.team_alias
                                ? `${team.team_alias.slice(0, 30)}...`
                                : `Team ${team.team_id.slice(0, 30)}...`}
                            </span>
                          </div>
                        </SelectItem>
                      ))}
                  </Select>
                </div>

                <div className="flex items-center gap-4">
                  <Text className="text-lg font-semibold text-gray-900">View:</Text>
                  <Select
                    className="w-64"
                    defaultValue="current_team"
                    value={modelViewMode}
                    onValueChange={(value) => setModelViewMode(value as "current_team" | "all")}
                  >
                    <SelectItem value="current_team">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                        <span className="font-medium">Current Team Models</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="all">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                        <span className="font-medium">All Available Models</span>
                      </div>
                    </SelectItem>
                  </Select>
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
                        {currentTeam}&quot; on the{" "}
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
                        value={selectedModelGroup ?? "all"}
                        onValueChange={(value) => setSelectedModelGroup(value === "all" ? "all" : value)}
                        placeholder="Filter by Public Model Name"
                      >
                        <SelectItem value="all">All Models</SelectItem>
                        <SelectItem value="wildcard">Wildcard Models (*)</SelectItem>
                        {availableModelGroups.map((group, idx) => (
                          <SelectItem key={idx} value={group}>
                            {group}
                          </SelectItem>
                        ))}
                      </Select>
                    </div>

                    {/* Model Access Group Filter */}
                    <div className="w-64">
                      <Select
                        value={selectedModelAccessGroupFilter ?? "all"}
                        onValueChange={(value) => setSelectedModelAccessGroupFilter(value === "all" ? null : value)}
                        placeholder="Filter by Model Access Group"
                      >
                        <SelectItem value="all">All Model Access Groups</SelectItem>
                        {availableModelAccessGroups.map((accessGroup, idx) => (
                          <SelectItem key={idx} value={accessGroup}>
                            {accessGroup}
                          </SelectItem>
                        ))}
                      </Select>
                    </div>
                  </div>
                )}

                {/* Results Count and Pagination Controls */}
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-700">
                    {filteredData.length > 0
                      ? `Showing ${pagination.pageIndex * pagination.pageSize + 1} - ${Math.min(
                          (pagination.pageIndex + 1) * pagination.pageSize,
                          filteredData.length,
                        )} of ${filteredData.length} results`
                      : "Showing 0 results"}
                  </span>

                  {/* Pagination Controls */}
                  {filteredData.length > pagination.pageSize && (
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() =>
                          setPagination((prev: PaginationState) => ({ ...prev, pageIndex: prev.pageIndex - 1 }))
                        }
                        disabled={pagination.pageIndex === 0}
                        className={`px-3 py-1 text-sm border rounded-md ${
                          pagination.pageIndex === 0
                            ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                            : "hover:bg-gray-50"
                        }`}
                      >
                        Previous
                      </button>

                      <button
                        onClick={() =>
                          setPagination((prev: PaginationState) => ({ ...prev, pageIndex: prev.pageIndex + 1 }))
                        }
                        disabled={pagination.pageIndex >= Math.ceil(filteredData.length / pagination.pageSize) - 1}
                        className={`px-3 py-1 text-sm border rounded-md ${
                          pagination.pageIndex >= Math.ceil(filteredData.length / pagination.pageSize) - 1
                            ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                            : "hover:bg-gray-50"
                        }`}
                      >
                        Next
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <ModelDataTable
              columns={columns(
                userRole,
                userId,
                premiumUser,
                setSelectedModelId,
                setSelectedTeamId,
                getDisplayModelName,
                () => {},
                () => {},
                setEditModel,
                expandedRows,
                setExpandedRows,
              )}
              data={paginatedData}
              isLoading={false}
              table={tableRef}
            />
          </div>
        </div>
      </Grid>
    </TabPanel>
  );
};

export default AllModelsTab;
