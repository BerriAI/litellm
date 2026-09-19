"use client";

import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import ModelSettingsModal from "@/components/model_dashboard/ModelSettingsModal/ModelSettingsModal";
import { ModelData } from "@/components/model_dashboard/types";
import { useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { toast } from "@/lib/toast";
import { uiHref } from "@/utils/uiHref";
import { modelDeleteCall, modelPatchUpdateCall, type Team } from "@/components/networking";
import { useQueryClient } from "@tanstack/react-query";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnFiltersState, functionalUpdate, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Info } from "lucide-react";
import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";

import { useModelsInfo } from "../../hooks/models/useModels";
import { transformModelData } from "../utils/modelDataTransformer";
import {
  ALL_MODEL_GROUPS_VALUE,
  AllModelsTable,
  MODEL_VIEW_MODES,
  ModelViewMode,
  PAGE_SIZE_OPTIONS,
  PERSONAL_TEAM_VALUE,
  WILDCARD_MODEL_GROUP_VALUE,
} from "./AllModelsTable";
import {
  ACCESS_GROUPS_COLUMN_ID,
  MODEL_NAME_COLUMN_ID,
  MODEL_TABLE_SORT_COLUMN_IDS,
  toServerSortField,
} from "./ModelsTableColumns";

const SEARCH_DEBOUNCE_WAIT_MS = 200;
const DEFAULT_PAGE_SIZE = 50;
const MAX_PAGE_SIZE = Math.max(...PAGE_SIZE_OPTIONS);
const DEFAULT_PAGINATION: PaginationState = { pageIndex: 0, pageSize: DEFAULT_PAGE_SIZE };
const UNSORTED_COLUMN_ID = "";
const FILTER_COLUMNS = [ACCESS_GROUPS_COLUMN_ID] as const;
type FilterColumn = (typeof FILTER_COLUMNS)[number];

const TABLE_STATE_OPTIONS: UrlTableStateOptions<FilterColumn> = {
  sortFields: MODEL_TABLE_SORT_COLUMN_IDS,
  defaultSort: { id: UNSORTED_COLUMN_ID, desc: false },
  defaultPageSize: DEFAULT_PAGE_SIZE,
  maxPageSize: MAX_PAGE_SIZE,
  filterColumns: FILTER_COLUMNS,
  urlKeys: { search: "model_search", filter_model_info_access_groups: "access_group" },
};

const SCOPE_PARSERS = {
  filter_team: parseAsString.withDefault(PERSONAL_TEAM_VALUE),
  view_mode: parseAsStringLiteral(MODEL_VIEW_MODES).withDefault("current_team"),
};

const toOfferedPageSize = (pageSize: number): number =>
  PAGE_SIZE_OPTIONS.includes(pageSize) ? pageSize : DEFAULT_PAGE_SIZE;

const appliedAccessGroup = (filters: ColumnFiltersState): string | null => {
  const value = filters.find((filter) => filter.id === ACCESS_GROUPS_COLUMN_ID)?.value;
  return typeof value === "string" ? value : null;
};

interface ModelGroupQuery {
  modelName?: string;
  wildcardOnly: boolean;
}

const toModelGroupQuery = (modelGroup: string | null): ModelGroupQuery => {
  if (!modelGroup || modelGroup === ALL_MODEL_GROUPS_VALUE) return { wildcardOnly: false };
  if (modelGroup === WILDCARD_MODEL_GROUP_VALUE) return { wildcardOnly: true };
  return { modelName: modelGroup, wildcardOnly: false };
};

const toAccessGroupQuery = (accessGroup: string | null): string | undefined =>
  accessGroup && accessGroup !== ALL_MODEL_GROUPS_VALUE ? accessGroup : undefined;

const useTeamScope = (teams: Team[] | undefined, isLoadingTeams: boolean) => {
  const [{ filter_team: urlTeamValue, view_mode }, setScope] = useQueryStates(SCOPE_PARSERS);
  const isKnownTeam =
    urlTeamValue === PERSONAL_TEAM_VALUE || (teams ?? []).some((team) => team.team_id === urlTeamValue);

  useEffect(() => {
    if (teams !== undefined && !isKnownTeam) void setScope({ filter_team: null });
  }, [teams, isKnownTeam, setScope]);

  return {
    selectedTeamValue: isKnownTeam ? urlTeamValue : PERSONAL_TEAM_VALUE,
    isTeamPending: isLoadingTeams && !isKnownTeam,
    modelViewMode: view_mode,
    setScope,
  };
};

interface ServerSort {
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

const toServerSortOrder = (sort: SortingState[number]): "asc" | "desc" => (sort.desc ? "desc" : "asc");

const toServerSort = (sort: SortingState[number] | undefined): ServerSort =>
  sort ? { sortBy: toServerSortField(sort.id), sortOrder: toServerSortOrder(sort) } : {};

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
  const { accessToken, userId, userRole, isViewOnly } = useAuthorized();
  const { data: teams, isLoading: isLoadingTeams } = useTeams();
  const queryClient = useQueryClient();

  const {
    search: modelNameSearch,
    setSearch,
    sorting: urlSorting,
    onSortingChange,
    pagination: urlPagination,
    onPaginationChange: onUrlPaginationChange,
    columnFilters: urlColumnFilters,
    onColumnFiltersChange,
  } = useUrlTableState(TABLE_STATE_OPTIONS);
  const { selectedTeamValue, isTeamPending, modelViewMode, setScope } = useTeamScope(teams, isLoadingTeams);
  const [debouncedSearch] = useDebouncedValue(modelNameSearch, { wait: SEARCH_DEBOUNCE_WAIT_MS });
  const [isModelSettingsModalVisible, setIsModelSettingsModalVisible] = useState(false);
  const [deleteModalModelId, setDeleteModalModelId] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [pausingModelId, setPausingModelId] = useState<string | null>(null);

  const offeredPageSize = toOfferedPageSize(urlPagination.pageSize);
  const pagination = useMemo<PaginationState>(
    () => ({ pageIndex: urlPagination.pageIndex, pageSize: offeredPageSize }),
    [urlPagination.pageIndex, offeredPageSize],
  );

  const activeSort = urlSorting.find((entry) => entry.id !== UNSORTED_COLUMN_ID);
  const sorting = useMemo<SortingState>(() => (activeSort ? [activeSort] : []), [activeSort]);
  const { sortBy, sortOrder } = toServerSort(activeSort);
  const teamIdForQuery = selectedTeamValue === PERSONAL_TEAM_VALUE ? undefined : selectedTeamValue;
  const { modelName: modelNameForQuery, wildcardOnly: wildcardOnlyForQuery } = toModelGroupQuery(selectedModelGroup);
  const accessGroupForQuery = toAccessGroupQuery(appliedAccessGroup(urlColumnFilters));

  const {
    data: rawModelData,
    isLoading: isLoadingModelsInfo,
    isFetching: isFetchingModelsInfo,
    isError: isErrorModelsInfo,
    refetch: refetchModels,
  } = useModelsInfo(
    pagination.pageIndex + 1,
    pagination.pageSize,
    debouncedSearch || undefined,
    undefined,
    teamIdForQuery,
    sortBy,
    sortOrder,
    // Auto-routers are routing constructs, not deployments; the sibling Auto-Routers tab
    // lists and manages them. Excluded server-side so total_count stays honest.
    true,
    modelNameForQuery,
    accessGroupForQuery,
    wildcardOnlyForQuery,
  );
  const isLoading = isLoadingModelsInfo || isLoadingModelCostMap || isTeamPending;

  const getProviderFromModel = useCallback(
    (model: string) => {
      if (modelCostMapData !== null && modelCostMapData !== undefined) {
        if (typeof modelCostMapData == "object" && model in modelCostMapData) {
          return modelCostMapData[model]["litellm_provider"];
        }
      }
      return "openai";
    },
    [modelCostMapData],
  );

  const modelData = useMemo<{ data: ModelData[] }>(() => {
    if (!rawModelData) return { data: [] };
    return transformModelData(rawModelData, getProviderFromModel);
  }, [rawModelData, getProviderFromModel]);

  const columnFilters = useMemo<ColumnFiltersState>(
    () => [
      ...(selectedModelGroup && selectedModelGroup !== ALL_MODEL_GROUPS_VALUE
        ? [{ id: MODEL_NAME_COLUMN_ID, value: selectedModelGroup }]
        : []),
      ...urlColumnFilters,
    ],
    [selectedModelGroup, urlColumnFilters],
  );

  const handleColumnFiltersChange: OnChangeFn<ColumnFiltersState> = (updater) => {
    const next = functionalUpdate(updater, columnFilters);
    const modelGroup = next.find((entry) => entry.id === MODEL_NAME_COLUMN_ID)?.value;
    setSelectedModelGroup(typeof modelGroup === "string" ? modelGroup : ALL_MODEL_GROUPS_VALUE);
    onColumnFiltersChange(next);
  };

  const handlePaginationChange: OnChangeFn<PaginationState> = (updater) => {
    onUrlPaginationChange(functionalUpdate(updater, pagination));
  };

  const handleSortingChange: OnChangeFn<SortingState> = (updater) => {
    onSortingChange(functionalUpdate(updater, sorting));
  };

  const handleTeamChange = (value: string) => {
    void setScope({ filter_team: value });
    onUrlPaginationChange({ pageIndex: 0, pageSize: pagination.pageSize });
  };

  const handleViewModeChange = (viewMode: ModelViewMode) => {
    void setScope({ view_mode: viewMode });
  };

  const resetFilters = () => {
    setSearch("");
    setSelectedModelGroup(ALL_MODEL_GROUPS_VALUE);
    onColumnFiltersChange([]);
    onSortingChange([]);
    onUrlPaginationChange(DEFAULT_PAGINATION);
    void setScope(null);
  };

  const teamOptions = useMemo(
    () => [
      { value: PERSONAL_TEAM_VALUE, label: "Personal" },
      ...(teams ?? [])
        .filter((team) => team.team_id)
        .map((team) => ({ value: team.team_id, label: team.team_alias ? team.team_alias : team.team_id })),
    ],
    [teams],
  );

  const selectedTeam = useMemo(
    () => (teams ?? []).find((team) => team.team_id === selectedTeamValue) ?? null,
    [teams, selectedTeamValue],
  );

  const modelToDelete = useMemo(() => {
    if (!deleteModalModelId || !modelData?.data) return null;
    return modelData.data.find((model: ModelData) => model.model_info.id === deleteModalModelId);
  }, [deleteModalModelId, modelData]);

  const handleDeleteModel = async () => {
    if (!accessToken || !deleteModalModelId) return;
    try {
      setDeleteLoading(true);
      await modelDeleteCall(accessToken, deleteModalModelId);
      toast.success("Model deleted successfully");
      queryClient.invalidateQueries({ queryKey: ["models", "list"] });
      refetchModels();
    } catch (error) {
      console.error("Error deleting model:", error);
      toast.fromError(error);
    } finally {
      setDeleteLoading(false);
      setDeleteModalModelId(null);
    }
  };

  const handleTogglePause = useCallback(
    async (modelId: string, blocked: boolean) => {
      if (!accessToken) return;
      try {
        setPausingModelId(modelId);
        await modelPatchUpdateCall(accessToken, { blocked }, modelId);
        toast.success(blocked ? "Model paused" : "Model resumed");
        // invalidateQueries already schedules a refetch for active observers
        // on this key — no need to also call refetchModels() (would double-fetch).
        queryClient.invalidateQueries({ queryKey: ["models", "list"] });
      } catch (error) {
        console.error("Error toggling model pause state:", error);
        toast.fromError(error);
      } finally {
        setPausingModelId(null);
      }
    },
    [accessToken, queryClient],
  );

  const handleRefresh = useCallback(() => {
    void refetchModels();
  }, [refetchModels]);

  const handleDeleteClick = useCallback((modelId: string) => {
    setDeleteModalModelId(modelId);
  }, []);

  const handleOpenModelSettings = useCallback(() => {
    setIsModelSettingsModalVisible(true);
  }, []);

  const teamAccessLabel = selectedTeam?.team_alias || selectedTeam?.team_id || "";

  return (
    <div className="w-full">
      <div className="flex flex-col gap-3">
        <AllModelsTable
          data={modelData.data}
          rowCount={rawModelData?.total_count ?? 0}
          isLoading={isLoading}
          isError={isErrorModelsInfo}
          isRefreshing={isFetchingModelsInfo}
          onRefresh={handleRefresh}
          sorting={sorting}
          onSortingChange={handleSortingChange}
          pagination={pagination}
          onPaginationChange={handlePaginationChange}
          columnFilters={columnFilters}
          onColumnFiltersChange={handleColumnFiltersChange}
          onResetFilters={resetFilters}
          searchValue={modelNameSearch}
          onSearchChange={setSearch}
          teamOptions={teamOptions}
          selectedTeamValue={selectedTeamValue}
          onTeamChange={handleTeamChange}
          isLoadingTeams={isLoadingTeams}
          viewMode={modelViewMode}
          onViewModeChange={handleViewModeChange}
          onOpenModelSettings={handleOpenModelSettings}
          availableModelGroups={availableModelGroups}
          availableModelAccessGroups={availableModelAccessGroups}
          userRole={userRole}
          userID={userId}
          isViewOnly={isViewOnly}
          onModelIdClick={setSelectedModelId}
          onTeamIdClick={setSelectedTeamId}
          onDeleteClick={handleDeleteClick}
          onTogglePauseClick={handleTogglePause}
          pausingModelId={pausingModelId}
        />

        {modelViewMode === "current_team" && (
          <div className="flex items-start gap-2 px-1 text-xs text-muted-foreground">
            <Info className="mt-0.5 size-3.5 shrink-0" />
            {selectedTeamValue === PERSONAL_TEAM_VALUE ? (
              <span>
                To access these models, create a Virtual Key without selecting a team on the{" "}
                <a href={uiHref("api-keys")} className="font-medium text-info hover:underline">
                  Virtual Keys page
                </a>
                .
              </span>
            ) : (
              <span>
                To access these models, create a Virtual Key and select Team as &quot;{teamAccessLabel}&quot; on the{" "}
                <a href={uiHref("api-keys")} className="font-medium text-info hover:underline">
                  Virtual Keys page
                </a>
                .
              </span>
            )}
          </div>
        )}
      </div>

      <DeleteResourceModal
        isOpen={!!deleteModalModelId}
        title="Delete Model"
        alertMessage="This action cannot be undone."
        message="Are you sure you want to delete this model?"
        resourceInformationTitle="Model Information"
        resourceInformation={
          modelToDelete
            ? [
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
              ]
            : []
        }
        onCancel={() => setDeleteModalModelId(null)}
        onOk={handleDeleteModel}
        confirmLoading={deleteLoading}
      />
      <ModelSettingsModal
        isVisible={isModelSettingsModalVisible}
        onCancel={() => setIsModelSettingsModalVisible(false)}
        onSuccess={() => setIsModelSettingsModalVisible(false)}
      />
    </div>
  );
};

export default AllModelsTab;
