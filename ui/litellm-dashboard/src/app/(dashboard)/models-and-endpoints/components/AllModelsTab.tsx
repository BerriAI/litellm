import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Team } from "@/components/key_team_helpers/key_list";
import { AllModelsDataTable } from "@/components/model_dashboard/all_models_table";
import { columns } from "@/components/molecules/models/columns";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { modelDeleteCall } from "@/components/networking";
import { Info, Settings, Filter, RefreshCw, Search } from "lucide-react";
import { PaginationState, SortingState } from "@tanstack/react-table";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import ModelSettingsModal from "@/components/model_dashboard/ModelSettingsModal/ModelSettingsModal";
import debounce from "lodash/debounce";
import { useEffect, useMemo, useState } from "react";
import { useModelsInfo } from "../../hooks/models/useModels";
import { transformModelData } from "../utils/modelDataTransformer";

type ModelViewMode = "all" | "current_team";

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
  const { accessToken, userId, userRole, premiumUser } = useAuthorized();
  const { data: teams, isLoading: isLoadingTeams } = useTeams();
  const queryClient = useQueryClient();

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
  const [isModelSettingsModalVisible, setIsModelSettingsModalVisible] = useState(false);

  const debouncedUpdateSearch = useMemo(
    () =>
      debounce((value: string) => {
        setDebouncedSearch(value);
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

  const teamIdForQuery = currentTeam === "personal" ? undefined : currentTeam.team_id;

  const sortBy = useMemo(() => {
    if (sorting.length === 0) return undefined;
    const sort = sorting[0];
    const columnIdToServerField: Record<string, string> = {
      input_cost: "costs",
      model_info_db_model: "status",
      model_info_created_by: "created_at",
      model_info_updated_at: "updated_at",
    };
    return columnIdToServerField[sort.id] || sort.id;
  }, [sorting]);

  const sortOrder = useMemo(() => {
    if (sorting.length === 0) return undefined;
    const sort = sorting[0];
    return sort.desc ? "desc" : "asc";
  }, [sorting]);

  const { data: rawModelData, isLoading: isLoadingModelsInfo, refetch: refetchModels } = useModelsInfo(
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

  const [deleteModalModelId, setDeleteModalModelId] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

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

      return modelNameMatch && accessGroupMatch;
    });
  }, [modelData, selectedModelGroup, selectedModelAccessGroupFilter]);

  useEffect(() => {
    setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
    setCurrentPage(1);
  }, [selectedModelGroup, selectedModelAccessGroupFilter]);

  useEffect(() => {
    setCurrentPage(1);
    setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
  }, [teamIdForQuery]);

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

  const modelToDelete = useMemo(() => {
    if (!deleteModalModelId || !modelData?.data) return null;
    return modelData.data.find((model: any) => model.model_info.id === deleteModalModelId);
  }, [deleteModalModelId, modelData]);

  const handleDeleteModel = async () => {
    if (!accessToken || !deleteModalModelId) return;
    try {
      setDeleteLoading(true);
      await modelDeleteCall(accessToken, deleteModalModelId);
      NotificationsManager.success("Model deleted successfully");
      queryClient.invalidateQueries({ queryKey: ["models", "list"] });
      refetchModels();
    } catch (error) {
      console.error("Error deleting model:", error);
      NotificationsManager.fromBackend(error);
    } finally {
      setDeleteLoading(false);
      setDeleteModalModelId(null);
    }
  };

  const currentTeamValue =
    currentTeam === "personal" ? "personal" : currentTeam.team_id;

  const onTeamChange = (value: string) => {
    if (value === "personal") {
      setCurrentTeam("personal");
      setCurrentPage(1);
      setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
    } else {
      const team = teams?.find((t) => t.team_id === value);
      if (team) {
        setCurrentTeam(team);
        setCurrentPage(1);
        setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
      }
    }
  };

  return (
    <>
      <div>
        <div className="flex flex-col space-y-4">
          <div className="bg-background rounded-lg shadow">
            {/* Current Team and View Mode Selector - Prominent Section */}
            <div className="border-b px-6 py-4 bg-muted/40">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <span className="text-lg font-semibold">Current Team:</span>
                  <div className="w-80">
                    {isLoading || isLoadingTeams ? (
                      <Skeleton className="h-11 w-full" />
                    ) : (
                      <Select value={currentTeamValue} onValueChange={onTeamChange}>
                        <SelectTrigger className="h-11">
                          <SelectValue placeholder="Select team" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="personal">
                            <span className="inline-flex items-center gap-2">
                              <Badge variant="secondary" className="h-2 w-2 p-0 rounded-full bg-blue-500" />
                              <span>Personal</span>
                            </span>
                          </SelectItem>
                          {teams
                            ?.filter((team) => team.team_id)
                            .map((team) => (
                              <SelectItem key={team.team_id} value={team.team_id}>
                                <span className="inline-flex items-center gap-2">
                                  <Badge variant="secondary" className="h-2 w-2 p-0 rounded-full bg-emerald-500" />
                                  <span className="truncate">
                                    {team.team_alias ? team.team_alias : team.team_id}
                                  </span>
                                </span>
                              </SelectItem>
                            ))}
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-lg font-semibold">View:</span>
                  <div className="w-64">
                    {isLoading ? (
                      <Skeleton className="h-11 w-full" />
                    ) : (
                      <Select
                        value={modelViewMode}
                        onValueChange={(value) =>
                          setModelViewMode(value as "current_team" | "all")
                        }
                      >
                        <SelectTrigger className="h-11">
                          <SelectValue placeholder="View" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="current_team">
                            <span className="inline-flex items-center gap-2">
                              <Badge variant="secondary" className="h-2 w-2 p-0 rounded-full bg-purple-500" />
                              <span>Current Team Models</span>
                            </span>
                          </SelectItem>
                          <SelectItem value="all">
                            <span className="inline-flex items-center gap-2">
                              <Badge variant="secondary" className="h-2 w-2 p-0 rounded-full bg-muted-foreground" />
                              <span>All Available Models</span>
                            </span>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                </div>
              </div>

              {modelViewMode === "current_team" && (
                <div className="flex items-start gap-2 mt-3">
                  <Info className="text-muted-foreground mt-0.5 flex-shrink-0 h-3 w-3" />
                  <div className="text-xs text-muted-foreground">
                    {currentTeam === "personal" ? (
                      <span>
                        To access these models: Create a Virtual Key without selecting a team on the{" "}
                        <a
                          href="/public?login=success&page=api-keys"
                          className="text-muted-foreground hover:text-foreground underline"
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
                          className="text-muted-foreground hover:text-foreground underline"
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
                <div className="flex items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-3">
                    {/* Model Name Search */}
                    <div className="relative w-64">
                      <input
                        type="text"
                        placeholder="Search model names..."
                        data-testid="model-search-input"
                        className="w-full px-3 py-2 pl-8 border border-input bg-background rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:border-ring"
                        value={modelNameSearch}
                        onChange={(e) => setModelNameSearch(e.target.value)}
                      />
                      <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                    </div>

                    {/* Filter Button */}
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className={showFilters ? "bg-muted" : ""}
                      onClick={() => setShowFilters(!showFilters)}
                    >
                      <Filter className="h-4 w-4 mr-2" />
                      Filters
                    </Button>

                    {/* Reset Filters Button */}
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={resetFilters}
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Reset Filters
                    </Button>
                  </div>

                  {/* Model Settings Button */}
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() => setIsModelSettingsModalVisible(true)}
                    title="Model Settings"
                    aria-label="Model Settings"
                  >
                    <Settings className="h-4 w-4" />
                  </Button>
                </div>

                {/* Additional Filters */}
                {showFilters && (
                  <div className="flex flex-wrap items-center gap-3 mt-3">
                    <div className="w-64">
                      <Select
                        value={selectedModelGroup ?? "all"}
                        onValueChange={(value) =>
                          setSelectedModelGroup(value === "all" ? "all" : value)
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Filter by Public Model Name" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Models</SelectItem>
                          <SelectItem value="wildcard">Wildcard Models (*)</SelectItem>
                          {availableModelGroups.map((group) => (
                            <SelectItem key={group} value={group}>
                              {group}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="w-64">
                      <Select
                        value={selectedModelAccessGroupFilter ?? "all"}
                        onValueChange={(value) =>
                          setSelectedModelAccessGroupFilter(value === "all" ? null : value)
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Filter by Model Access Group" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Model Access Groups</SelectItem>
                          {availableModelAccessGroups.map((accessGroup) => (
                            <SelectItem key={accessGroup} value={accessGroup}>
                              {accessGroup}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                )}

                {/* Results Count and Pagination Controls */}
                <div className="flex justify-between items-center">
                  {isLoading ? (
                    <Skeleton className="h-5 w-[184px]" />
                  ) : (
                    <span data-testid="models-results-count" className="text-sm text-foreground">
                      {paginationMeta.total_count > 0
                        ? `Showing ${((currentPage - 1) * pageSize) + 1} - ${Math.min(currentPage * pageSize, paginationMeta.total_count)} of ${paginationMeta.total_count} results`
                        : "Showing 0 results"}
                    </span>
                  )}

                  <div className="flex items-center space-x-2">
                    {isLoading ? (
                      <Skeleton className="h-8 w-[84px]" />
                    ) : (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const newPage = currentPage - 1;
                          setCurrentPage(newPage);
                          setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
                        }}
                        disabled={currentPage === 1}
                      >
                        Previous
                      </Button>
                    )}

                    {isLoading ? (
                      <Skeleton className="h-8 w-[56px]" />
                    ) : (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const newPage = currentPage + 1;
                          setCurrentPage(newPage);
                          setPagination((prev: PaginationState) => ({ ...prev, pageIndex: 0 }));
                        }}
                        disabled={currentPage >= paginationMeta.total_pages}
                      >
                        Next
                      </Button>
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
                setDeleteModalModelId,
              )}
              data={filteredData}
              isLoading={isLoadingModelsInfo}
              sorting={sorting}
              onSortingChange={setSorting}
              pagination={pagination}
              onPaginationChange={setPagination}
              enablePagination={true}
              onRowClick={(model: any) => setSelectedModelId(model.model_info.id)}
            />
          </div>
        </div>
      </div>

      <DeleteResourceModal
        isOpen={!!deleteModalModelId}
        title="Delete Model"
        alertMessage="This action cannot be undone."
        message="Are you sure you want to delete this model?"
        resourceInformationTitle="Model Information"
        resourceInformation={modelToDelete ? [
          {
            label: "Model Name",
            value: modelToDelete.model_name || "Not Set",
          },
          {
            label: "LiteLLM Model Name",
            value: modelToDelete.litellm_model_name || "Not Set",
          },
          {
            label: "Provider",
            value: modelToDelete.provider || "Not Set",
          },
          {
            label: "Created By",
            value: modelToDelete.model_info?.created_by || "Not Set",
          },
        ] : []}
        onCancel={() => setDeleteModalModelId(null)}
        onOk={handleDeleteModel}
        confirmLoading={deleteLoading}
      />
      <ModelSettingsModal
        isVisible={isModelSettingsModalVisible}
        onCancel={() => setIsModelSettingsModalVisible(false)}
        onSuccess={() => setIsModelSettingsModalVisible(false)}
      />
    </>
  );
};

export default AllModelsTab;
