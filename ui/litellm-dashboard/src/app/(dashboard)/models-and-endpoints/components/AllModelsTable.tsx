"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { ROUTING_STATUS_FILTER_ID } from "../utils/routingStatus";

import { ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Search, Settings } from "lucide-react";
import { useMemo, useState } from "react";

import { ModelData } from "@/components/model_dashboard/types";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
} from "@/components/shared/DataTable";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import { ToolbarSeparator } from "@/components/shared/ToolbarSeparator";
import { cn } from "@/lib/cva.config";

import {
  ACCESS_GROUPS_COLUMN_ID,
  getModelsTableColumns,
  MODEL_NAME_COLUMN_ID,
  STATUS_COLUMN_ID,
} from "./ModelsTableColumns";

export type ModelViewMode = "all" | "current_team";

export const PERSONAL_TEAM_VALUE = "personal";
export const ALL_MODEL_GROUPS_VALUE = "all";
export const WILDCARD_MODEL_GROUP_VALUE = "wildcard";

const MODEL_TABLE_BODY_HEIGHT = 600;
const ALL_PROXY_MODELS_LABEL = "All Proxy Models";

const FILTER_LABELS: Record<string, string> = {
  [MODEL_NAME_COLUMN_ID]: "Public Model Name",
  [ACCESS_GROUPS_COLUMN_ID]: "Model Access Group",
  [ROUTING_STATUS_FILTER_ID]: "Status",
};

const ROUTING_STATUS_FILTER_VALUES = {
  all: "All",
  active: "Active",
  paused: "Paused",
} as const;

const VIEW_MODE_LABELS: Record<ModelViewMode, string> = {
  current_team: "Current Team Models",
  all: ALL_PROXY_MODELS_LABEL,
};

export interface ModelsTableTeamOption {
  value: string;
  label: string;
}

interface AllModelsTableProps {
  data: ModelData[];
  rowCount: number;
  isLoading: boolean;
  isRefreshing: boolean;
  onRefresh: () => void;
  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  columnFilters: ColumnFiltersState;
  onColumnFiltersChange: OnChangeFn<ColumnFiltersState>;
  onResetFilters: () => void;
  searchValue: string;
  onSearchChange: (value: string) => void;
  teamOptions: ModelsTableTeamOption[];
  selectedTeamValue: string;
  onTeamChange: (value: string) => void;
  isLoadingTeams: boolean;
  viewMode: ModelViewMode;
  onViewModeChange: (viewMode: ModelViewMode) => void;
  onOpenModelSettings: () => void;
  availableModelGroups: string[];
  availableModelAccessGroups: string[];
  userRole: string;
  userID: string;
  isViewOnly: boolean;
  onModelIdClick: (modelId: string) => void;
  onTeamIdClick: (teamId: string) => void;
  onDeleteClick: (modelId: string) => void;
  onTogglePauseClick: (modelId: string, blocked: boolean) => void | Promise<void>;
  pausingModelId: string | null;
  credentialLabels?: ReadonlyMap<string, string>;
}

function EmptyState() {
  const { t } = useLanguage();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-11 items-center justify-center rounded-xl bg-muted">
        <Search className="size-5 text-muted-foreground" />
      </div>
      <div className="text-base font-semibold text-foreground">{t("No models found")}</div>
      <div className="max-w-80 text-sm text-muted-foreground">
        {t("No models match your search or filters. Try resetting them.")}
      </div>
    </div>
  );
}

export function AllModelsTable({
  data,
  rowCount,
  isLoading,
  isRefreshing,
  onRefresh,
  sorting,
  onSortingChange,
  pagination,
  onPaginationChange,
  columnFilters,
  onColumnFiltersChange,
  onResetFilters,
  searchValue,
  onSearchChange,
  teamOptions,
  selectedTeamValue,
  onTeamChange,
  isLoadingTeams,
  viewMode,
  onViewModeChange,
  onOpenModelSettings,
  availableModelGroups,
  availableModelAccessGroups,
  userRole,
  userID,
  isViewOnly,
  onModelIdClick,
  onTeamIdClick,
  onDeleteClick,
  onTogglePauseClick,
  pausingModelId,
  credentialLabels,
}: AllModelsTableProps) {
  const { t } = useLanguage();
  const [filtersOpen, setFiltersOpen] = useState(false);

  const columns = useMemo(() => {
    const columnDeps = {
      t,
      userRole,
      userID,
      isViewOnly,
      onModelIdClick,
      onTeamIdClick,
      onDeleteClick,
      onTogglePauseClick,
      pausingModelId,
      credentialLabels,
    };
    return getModelsTableColumns(columnDeps);
  }, [
    userRole,
    userID,
    isViewOnly,
    onModelIdClick,
    onTeamIdClick,
    onDeleteClick,
    onTogglePauseClick,
    pausingModelId,
    credentialLabels,
    t,
  ]);

  const modelGroupOptions = useMemo(
    () => [
      { label: t(ALL_PROXY_MODELS_LABEL), value: ALL_MODEL_GROUPS_VALUE },
      { label: t("Wildcard Models (*)"), value: WILDCARD_MODEL_GROUP_VALUE },
      ...availableModelGroups.map((group) => ({ label: group, value: group })),
    ],
    [availableModelGroups, t],
  );

  const accessGroupOptions = useMemo(
    () => [
      { label: t("All Model Access Groups"), value: ALL_MODEL_GROUPS_VALUE },
      ...availableModelAccessGroups.map((accessGroup) => ({ label: accessGroup, value: accessGroup })),
    ],
    [availableModelAccessGroups, t],
  );

  const formatFilterValue = (columnId: string, value: unknown): string => {
    const raw = String(value);
    if (columnId === MODEL_NAME_COLUMN_ID && raw === WILDCARD_MODEL_GROUP_VALUE) {
      return t("Wildcard Models (*)");
    }
    if (columnId === ROUTING_STATUS_FILTER_ID) {
      const label = ROUTING_STATUS_FILTER_VALUES[raw as keyof typeof ROUTING_STATUS_FILTER_VALUES];
      return label ? t(label) : raw;
    }
    return raw;
  };

  const selectedTeamLabel =
    teamOptions.find((option) => option.value === selectedTeamValue)?.label ?? teamOptions[0]?.label ?? "";

  return (
    <DataTable
      data={data}
      columns={columns}
      getRowId={(row, index) => row.model_info?.id ?? String(index)}
      sortingMode="server"
      sorting={sorting}
      onSortingChange={onSortingChange}
      enableSortingRemoval
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={rowCount}
      pageSizeOptions={[10, 25, 50]}
      filterMode="server"
      columnFilters={columnFilters}
      onColumnFiltersChange={onColumnFiltersChange}
      defaultColumnVisibility={{ [STATUS_COLUMN_ID]: false }}
      enableColumnResizing
      maxBodyHeight={MODEL_TABLE_BODY_HEIGHT}
      isLoading={isLoading}
      loadingMessage={t("Loading models…")}
      noDataMessage={<EmptyState />}
      size="compact"
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchValue}
            onSearchChange={onSearchChange}
            searchPlaceholder={t("Search model names…")}
            onOpenFilters={() => setFiltersOpen(true)}
            onRefresh={onRefresh}
            isRefreshing={isRefreshing}
            filterLabels={Object.fromEntries(Object.entries(FILTER_LABELS).map(([id, label]) => [id, t(label)]))}
            formatFilterValue={formatFilterValue}
          >
            <Select value={selectedTeamValue} onValueChange={(value) => onTeamChange(String(value))}>
              <SelectTrigger
                size="sm"
                aria-label={t("Current team")}
                data-testid="models-team-select"
                className="gap-2 bg-secondary"
              >
                <span
                  className={cn(
                    "size-2 shrink-0 rounded-full",
                    selectedTeamValue === PERSONAL_TEAM_VALUE ? "bg-info" : "bg-success",
                  )}
                />
                <span className="text-muted-foreground">{t("Team")}</span>
                <span className="truncate font-semibold">{selectedTeamLabel}</span>
              </SelectTrigger>
              <SelectContent>
                {teamOptions.map((option) => (
                  <SelectItem
                    key={option.value}
                    value={option.value}
                    disabled={isLoadingTeams}
                    className="[&>div]:min-w-0"
                  >
                    <span data-slot="select-item-label" className="min-w-0 truncate" title={option.label}>
                      {option.label}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={viewMode} onValueChange={(value) => onViewModeChange(value as ModelViewMode)}>
              <SelectTrigger size="sm" aria-label={t("View")} data-testid="models-view-select" className="gap-2">
                <span className="text-muted-foreground">{t("View")}</span>
                <span className="truncate">{t(VIEW_MODE_LABELS[viewMode])}</span>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="current_team">{t(VIEW_MODE_LABELS.current_team)}</SelectItem>
                <SelectItem value="all">{t(VIEW_MODE_LABELS.all)}</SelectItem>
              </SelectContent>
            </Select>

            <ToolbarSeparator className="mx-0.5" />

            <Button
              variant="outline"
              size="icon-sm"
              aria-label={t("Model Settings")}
              title={t("Model Settings")}
              data-testid="models-settings-trigger"
              onClick={onOpenModelSettings}
            >
              <Settings />
            </Button>
          </DataTableToolbar>
          <DataTableFilterDrawer
            table={table}
            open={filtersOpen}
            onOpenChange={setFiltersOpen}
            title={t("Filters")}
            description={t("Narrow down models + endpoints")}
            resetLabel={t("Reset Filters")}
            onReset={onResetFilters}
          >
            {({ get, set }) => (
              <>
                <DataTableFilterField label={t("Public Model Name")}>
                  <SearchSelect
                    options={modelGroupOptions}
                    value={(get(MODEL_NAME_COLUMN_ID) as string) ?? ALL_MODEL_GROUPS_VALUE}
                    onValueChange={(value) =>
                      set(MODEL_NAME_COLUMN_ID, value === ALL_MODEL_GROUPS_VALUE ? undefined : value ?? undefined)
                    }
                    placeholder={t("Filter by Public Model Name")}
                    emptyText={t("No models found")}
                  />
                </DataTableFilterField>
                <DataTableFilterField label={t("Status")}>
                  <Select
                    value={(get(ROUTING_STATUS_FILTER_ID) as string) ?? "all"}
                    onValueChange={(value) =>
                      set(ROUTING_STATUS_FILTER_ID, value === "all" ? undefined : (value as "active" | "paused"))
                    }
                  >
                    <SelectTrigger aria-label={t("Filter by status")} className="w-full">
                      {t(
                        ROUTING_STATUS_FILTER_VALUES[
                          (get(ROUTING_STATUS_FILTER_ID) as keyof typeof ROUTING_STATUS_FILTER_VALUES) ?? "all"
                        ] ?? ROUTING_STATUS_FILTER_VALUES.all,
                      )}
                    </SelectTrigger>
                    <SelectContent>
                      {(
                        Object.keys(ROUTING_STATUS_FILTER_VALUES) as Array<keyof typeof ROUTING_STATUS_FILTER_VALUES>
                      ).map((value) => (
                        <SelectItem key={value} value={value}>
                          {t(ROUTING_STATUS_FILTER_VALUES[value])}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </DataTableFilterField>
                <DataTableFilterField label={t("Model Access Group")}>
                  <SearchSelect
                    options={accessGroupOptions}
                    value={(get(ACCESS_GROUPS_COLUMN_ID) as string) ?? ALL_MODEL_GROUPS_VALUE}
                    onValueChange={(value) =>
                      set(ACCESS_GROUPS_COLUMN_ID, value === ALL_MODEL_GROUPS_VALUE ? undefined : value ?? undefined)
                    }
                    placeholder={t("Filter by Model Access Group")}
                    emptyText={t("No model access groups found")}
                  />
                </DataTableFilterField>
              </>
            )}
          </DataTableFilterDrawer>
        </>
      )}
    />
  );
}
