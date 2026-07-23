"use client";

import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import ModelSettingsModal from "@/components/model_dashboard/ModelSettingsModal/ModelSettingsModal";
import { ModelData } from "@/components/model_dashboard/types";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { modelDeleteCall, modelPatchUpdateCall } from "@/components/networking";
import { useQueryClient } from "@tanstack/react-query";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Info } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { useModelsInfo } from "../../hooks/models/useModels";
import { transformModelData } from "../utils/modelDataTransformer";
import {
  ALL_MODEL_GROUPS_VALUE,
  AllModelsTable,
  ModelViewMode,
  PERSONAL_TEAM_VALUE,
  WILDCARD_MODEL_GROUP_VALUE,
} from "./AllModelsTable";
import { ACCESS_GROUPS_COLUMN_ID, MODEL_NAME_COLUMN_ID, toServerSortField } from "./ModelsTableColumns";

const SEARCH_DEBOUNCE_WAIT_MS = 200;
const DEFAULT_PAGE_SIZE = 50;
const DEFAULT_PAGINATION: PaginationState = { pageIndex: 0, pageSize: DEFAULT_PAGE_SIZE };

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
  const { accessToken, userId, userRole } = useAuthorized();
  const { data: teams, isLoading: isLoadingTeams } = useTeams();
  const queryClient = useQueryClient();

  const [modelNameSearch, setModelNameSearch] = useState<string>("");
  const [debouncedSearch, setDebouncedSearch] = useState<string>("");
  const [modelViewMode, setModelViewMode] = useState<ModelViewMode>("current_team");
  const [selectedTeamValue, setSelectedTeamValue] = useState<string>(PERSONAL_TEAM_VALUE);
  const [selectedModelAccessGroupFilter, setSelectedModelAccessGroupFilter] = useState<string | null>(null);
  const [pagination, setPagination] = useState<PaginationState>(DEFAULT_PAGINATION);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [isModelSettingsModalVisible, setIsModelSettingsModalVisible] = useState(false);
  const [deleteModalModelId, setDeleteModalModelId] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [pausingModelId, setPausingModelId] = useState<string | null>(null);

  const resetToFirstPage = useCallback(() => {
    setPagination((previous) => ({ ...previous, pageIndex: 0 }));
  }, []);

  const debouncedUpdateSearch = useDebouncedCallback(
    (value: string) => {
      setDebouncedSearch(value);
      resetToFirstPage();
    },
    { wait: SEARCH_DEBOUNCE_WAIT_MS },
  );

  useEffect(() => {
    debouncedUpdateSearch(modelNameSearch);
  }, [modelNameSearch, debouncedUpdateSearch]);

  const teamIdForQuery = selectedTeamValue === PERSONAL_TEAM_VALUE ? undefined : selectedTeamValue;

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

  const modelData = useMemo(() => {
    if (!rawModelData) return { data: [] };
    return transformModelData(rawModelData, getProviderFromModel);
  }, [rawModelData, getProviderFromModel]);

  const filteredData = useMemo<ModelData[]>(() => {
    if (!modelData || !modelData.data || modelData.data.length === 0) {
      return [];
    }

    return modelData.data.filter((model: ModelData) => {
      const modelNameMatch =
        selectedModelGroup === ALL_MODEL_GROUPS_VALUE ||
        model.model_name === selectedModelGroup ||
        !selectedModelGroup ||
        (selectedModelGroup === WILDCARD_MODEL_GROUP_VALUE && model.model_name?.includes("*"));

      const accessGroupMatch =
        selectedModelAccessGroupFilter === ALL_MODEL_GROUPS_VALUE ||
        model.model_info["access_groups"]?.includes(selectedModelAccessGroupFilter ?? "") ||
        !selectedModelAccessGroupFilter;

      return modelNameMatch && accessGroupMatch;
    });
  }, [modelData, selectedModelGroup, selectedModelAccessGroupFilter]);

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

  const handleColumnFiltersChange: OnChangeFn<ColumnFiltersState> = (updater) => {
    const next = typeof updater === "function" ? updater(columnFilters) : updater;
    const modelGroup = next.find((entry) => entry.id === MODEL_NAME_COLUMN_ID)?.value;
    const accessGroup = next.find((entry) => entry.id === ACCESS_GROUPS_COLUMN_ID)?.value;
    setSelectedModelGroup(typeof modelGroup === "string" ? modelGroup : ALL_MODEL_GROUPS_VALUE);
    setSelectedModelAccessGroupFilter(typeof accessGroup === "string" ? accessGroup : null);
    resetToFirstPage();
  };

  const handleSortingChange: OnChangeFn<SortingState> = (updater) => {
    setSorting(typeof updater === "function" ? updater(sorting) : updater);
    resetToFirstPage();
  };

  const handleTeamChange = (value: string) => {
    setSelectedTeamValue(value);
    resetToFirstPage();
  };

  const resetFilters = () => {
    setModelNameSearch("");
    setSelectedModelGroup(ALL_MODEL_GROUPS_VALUE);
    setSelectedModelAccessGroupFilter(null);
    setSelectedTeamValue(PERSONAL_TEAM_VALUE);
    setModelViewMode("current_team");
    setPagination(DEFAULT_PAGINATION);
    setSorting([]);
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

  const handleTogglePause = useCallback(
    async (modelId: string, blocked: boolean) => {
      if (!accessToken) return;
      try {
        setPausingModelId(modelId);
        await modelPatchUpdateCall(accessToken, { blocked }, modelId);
        NotificationsManager.success(blocked ? "Model paused" : "Model resumed");
        // invalidateQueries already schedules a refetch for active observers
        // on this key — no need to also call refetchModels() (would double-fetch).
        queryClient.invalidateQueries({ queryKey: ["models", "list"] });
      } catch (error) {
        console.error("Error toggling model pause state:", error);
        NotificationsManager.fromBackend(error);
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
          data={filteredData}
          rowCount={rawModelData?.total_count ?? 0}
          isLoading={isLoading}
          isRefreshing={isFetchingModelsInfo}
          onRefresh={handleRefresh}
          sorting={sorting}
          onSortingChange={handleSortingChange}
          pagination={pagination}
          onPaginationChange={setPagination}
          columnFilters={columnFilters}
          onColumnFiltersChange={handleColumnFiltersChange}
          onResetFilters={resetFilters}
          searchValue={modelNameSearch}
          onSearchChange={setModelNameSearch}
          teamOptions={teamOptions}
          selectedTeamValue={selectedTeamValue}
          onTeamChange={handleTeamChange}
          isLoadingTeams={isLoadingTeams}
          viewMode={modelViewMode}
          onViewModeChange={setModelViewMode}
          onOpenModelSettings={handleOpenModelSettings}
          availableModelGroups={availableModelGroups}
          availableModelAccessGroups={availableModelAccessGroups}
          userRole={userRole}
          userID={userId}
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
                <a href="/public?login=success&page=api-keys" className="font-medium text-blue-600 hover:underline">
                  Virtual Keys page
                </a>
                .
              </span>
            ) : (
              <span>
                To access these models, create a Virtual Key and select Team as &quot;{teamAccessLabel}&quot; on the{" "}
                <a href="/public?login=success&page=api-keys" className="font-medium text-blue-600 hover:underline">
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
