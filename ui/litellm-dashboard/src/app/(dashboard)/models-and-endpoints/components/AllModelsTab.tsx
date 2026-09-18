"use client";

import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import ModelSettingsModal from "@/components/model_dashboard/ModelSettingsModal/ModelSettingsModal";
import { ModelData } from "@/components/model_dashboard/types";
import { toast } from "@/lib/toast";
import { uiHref } from "@/utils/uiHref";
import { modelDeleteCall, modelPatchUpdateCall } from "@/components/networking";
import { useQueryClient } from "@tanstack/react-query";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnFiltersState, functionalUpdate, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Info } from "lucide-react";
import { createParser, parseAsInteger, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useMemo, useState } from "react";

import { useModelsInfo } from "../../hooks/models/useModels";
import { transformModelData } from "../utils/modelDataTransformer";
import {
  ALL_MODEL_GROUPS_VALUE,
  AllModelsTable,
  ModelViewMode,
  PERSONAL_TEAM_VALUE,
  WILDCARD_MODEL_GROUP_VALUE,
} from "./AllModelsTable";
import {
  ACCESS_GROUPS_COLUMN_ID,
  isModelTableSortColumnId,
  MODEL_NAME_COLUMN_ID,
  MODEL_TABLE_SORT_COLUMN_IDS,
  toServerSortField,
} from "./ModelsTableColumns";

const SEARCH_DEBOUNCE_WAIT_MS = 200;
const DEFAULT_PAGE_SIZE = 50;
const MAX_PAGE_SIZE = 100;
const MAX_PAGE = 100_000;

const MODEL_VIEW_MODES = ["current_team", "all"] as const satisfies readonly ModelViewMode[];

const boundedInteger = (min: number, max: number, fallback: number) =>
  createParser({
    parse: (value: string) => {
      const parsed = parseAsInteger.parse(value);
      return parsed === null ? null : Math.min(Math.max(parsed, min), max);
    },
    serialize: String,
  }).withDefault(fallback);

const TABLE_STATE = {
  model_search: parseAsString.withDefault(""),
  view_mode: parseAsStringLiteral(MODEL_VIEW_MODES).withDefault("current_team"),
  filter_team: parseAsString.withDefault(PERSONAL_TEAM_VALUE),
  access_group: parseAsString.withDefault(""),
  sort_by: parseAsStringLiteral(MODEL_TABLE_SORT_COLUMN_IDS),
  sort_order: parseAsStringLiteral(["asc", "desc"] as const).withDefault("asc"),
  page: boundedInteger(1, MAX_PAGE, 1),
  page_size: boundedInteger(1, MAX_PAGE_SIZE, DEFAULT_PAGE_SIZE),
};

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

  const [tableState, setTableState] = useQueryStates(TABLE_STATE);
  const modelNameSearch = tableState.model_search;
  const [debouncedSearch] = useDebouncedValue(modelNameSearch, { wait: SEARCH_DEBOUNCE_WAIT_MS });
  const modelViewMode = tableState.view_mode;
  const selectedTeamValue = tableState.filter_team;
  const selectedModelAccessGroupFilter = tableState.access_group || null;
  const pagination = useMemo<PaginationState>(
    () => ({ pageIndex: tableState.page - 1, pageSize: tableState.page_size }),
    [tableState.page, tableState.page_size],
  );
  const sorting = useMemo<SortingState>(
    () => (tableState.sort_by ? [{ id: tableState.sort_by, desc: tableState.sort_order === "desc" }] : []),
    [tableState.sort_by, tableState.sort_order],
  );
  const [isModelSettingsModalVisible, setIsModelSettingsModalVisible] = useState(false);
  const [deleteModalModelId, setDeleteModalModelId] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [pausingModelId, setPausingModelId] = useState<string | null>(null);

  const teamIdForQuery = selectedTeamValue === PERSONAL_TEAM_VALUE ? undefined : selectedTeamValue;
  const isConcreteModelGroup =
    Boolean(selectedModelGroup) &&
    selectedModelGroup !== ALL_MODEL_GROUPS_VALUE &&
    selectedModelGroup !== WILDCARD_MODEL_GROUP_VALUE;
  const modelNameForQuery = isConcreteModelGroup ? selectedModelGroup ?? undefined : undefined;
  const accessGroupForQuery =
    selectedModelAccessGroupFilter && selectedModelAccessGroupFilter !== ALL_MODEL_GROUPS_VALUE
      ? selectedModelAccessGroupFilter
      : undefined;
  const wildcardOnlyForQuery = selectedModelGroup === WILDCARD_MODEL_GROUP_VALUE;

  const sortBy = useMemo(() => {
    if (sorting.length === 0) return undefined;
    return toServerSortField(sorting[0].id);
  }, [sorting]);

  const sortOrder = useMemo(() => {
    if (sorting.length === 0) return undefined;
    return sorting[0].desc ? "desc" : "asc";
  }, [sorting]);

  const {
    data: rawModelData,
    isLoading: isLoadingModelsInfo,
    isFetching: isFetchingModelsInfo,
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
  const isLoading = isLoadingModelsInfo || isLoadingModelCostMap;

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
    () =>
      [
        selectedModelGroup && selectedModelGroup !== ALL_MODEL_GROUPS_VALUE
          ? { id: MODEL_NAME_COLUMN_ID, value: selectedModelGroup }
          : null,
        selectedModelAccessGroupFilter ? { id: ACCESS_GROUPS_COLUMN_ID, value: selectedModelAccessGroupFilter } : null,
      ].filter((entry) => entry !== null),
    [selectedModelGroup, selectedModelAccessGroupFilter],
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      void setTableState({ model_search: value || null, page: null });
    },
    [setTableState],
  );

  const handleColumnFiltersChange: OnChangeFn<ColumnFiltersState> = (updater) => {
    const next = functionalUpdate(updater, columnFilters);
    const modelGroup = next.find((entry) => entry.id === MODEL_NAME_COLUMN_ID)?.value;
    const accessGroup = next.find((entry) => entry.id === ACCESS_GROUPS_COLUMN_ID)?.value;
    setSelectedModelGroup(typeof modelGroup === "string" ? modelGroup : ALL_MODEL_GROUPS_VALUE);
    void setTableState({ access_group: typeof accessGroup === "string" ? accessGroup : null, page: null });
  };

  const handleSortingChange: OnChangeFn<SortingState> = (updater) => {
    const active = functionalUpdate(updater, sorting)[0];
    void setTableState({
      sort_by: active && isModelTableSortColumnId(active.id) ? active.id : null,
      sort_order: active?.desc ? "desc" : null,
      page: null,
    });
  };

  const handlePaginationChange = useCallback<OnChangeFn<PaginationState>>(
    (updater) => {
      const next = functionalUpdate(updater, pagination);
      void setTableState({ page: next.pageIndex + 1, page_size: next.pageSize });
    },
    [pagination, setTableState],
  );

  const handleTeamChange = (value: string) => {
    void setTableState({ filter_team: value, page: null });
  };

  const handleViewModeChange = (value: ModelViewMode) => {
    void setTableState({ view_mode: value });
  };

  const resetFilters = () => {
    setSelectedModelGroup(ALL_MODEL_GROUPS_VALUE);
    void setTableState(null);
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
          onSearchChange={handleSearchChange}
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
