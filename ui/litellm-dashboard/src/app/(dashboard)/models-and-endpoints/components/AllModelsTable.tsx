"use client";

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
import { Separator } from "@/components/ui/separator";
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

const FILTER_LABELS: Record<string, string> = {
  [MODEL_NAME_COLUMN_ID]: "Public Model Name",
  [ACCESS_GROUPS_COLUMN_ID]: "Model Access Group",
};

const VIEW_MODE_LABELS: Record<ModelViewMode, string> = {
  current_team: "Current Team Models",
  all: "All Available Models",
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
  onModelIdClick: (modelId: string) => void;
  onTeamIdClick: (teamId: string) => void;
  onDeleteClick: (modelId: string) => void;
  onTogglePauseClick: (modelId: string, blocked: boolean) => void | Promise<void>;
  pausingModelId: string | null;
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-11 items-center justify-center rounded-xl bg-muted">
        <Search className="size-5 text-muted-foreground" />
      </div>
      <div className="text-base font-semibold text-foreground">No models found</div>
      <div className="max-w-80 text-sm text-muted-foreground">
        No models match your search or filters. Try resetting them.
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
  onModelIdClick,
  onTeamIdClick,
  onDeleteClick,
  onTogglePauseClick,
  pausingModelId,
}: AllModelsTableProps) {
  const [filtersOpen, setFiltersOpen] = useState(false);

  const columns = useMemo(() => {
    const columnDeps = {
      userRole,
      userID,
      onModelIdClick,
      onTeamIdClick,
      onDeleteClick,
      onTogglePauseClick,
      pausingModelId,
    };
    return getModelsTableColumns(columnDeps);
  }, [userRole, userID, onModelIdClick, onTeamIdClick, onDeleteClick, onTogglePauseClick, pausingModelId]);

  const modelGroupOptions = useMemo(
    () => [
      { label: "All Models", value: ALL_MODEL_GROUPS_VALUE },
      { label: "Wildcard Models (*)", value: WILDCARD_MODEL_GROUP_VALUE },
      ...availableModelGroups.map((group) => ({ label: group, value: group })),
    ],
    [availableModelGroups],
  );

  const accessGroupOptions = useMemo(
    () => [
      { label: "All Model Access Groups", value: ALL_MODEL_GROUPS_VALUE },
      ...availableModelAccessGroups.map((accessGroup) => ({ label: accessGroup, value: accessGroup })),
    ],
    [availableModelAccessGroups],
  );

  const formatFilterValue = (columnId: string, value: unknown): string => {
    const raw = String(value);
    if (columnId === MODEL_NAME_COLUMN_ID && raw === WILDCARD_MODEL_GROUP_VALUE) {
      return "Wildcard Models (*)";
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
      loadingMessage="Loading models…"
      noDataMessage={<EmptyState />}
      size="compact"
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchValue}
            onSearchChange={onSearchChange}
            searchPlaceholder="Search model names…"
            onOpenFilters={() => setFiltersOpen(true)}
            onRefresh={onRefresh}
            isRefreshing={isRefreshing}
            filterLabels={FILTER_LABELS}
            formatFilterValue={formatFilterValue}
          >
            <Select value={selectedTeamValue} onValueChange={(value) => onTeamChange(String(value))}>
              <SelectTrigger
                size="sm"
                aria-label="Current team"
                data-testid="models-team-select"
                className="gap-2 bg-secondary"
              >
                <span
                  className={cn(
                    "size-2 shrink-0 rounded-full",
                    selectedTeamValue === PERSONAL_TEAM_VALUE ? "bg-blue-500" : "bg-green-500",
                  )}
                />
                <span className="text-muted-foreground">Team</span>
                <span className="truncate font-semibold">{selectedTeamLabel}</span>
              </SelectTrigger>
              <SelectContent>
                {teamOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value} disabled={isLoadingTeams}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={viewMode} onValueChange={(value) => onViewModeChange(value as ModelViewMode)}>
              <SelectTrigger size="sm" aria-label="View" data-testid="models-view-select" className="gap-2">
                <span className="text-muted-foreground">View</span>
                <span className="truncate">{VIEW_MODE_LABELS[viewMode]}</span>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="current_team">{VIEW_MODE_LABELS.current_team}</SelectItem>
                <SelectItem value="all">{VIEW_MODE_LABELS.all}</SelectItem>
              </SelectContent>
            </Select>

            <Separator orientation="vertical" className="mx-0.5 h-5" />

            <Button
              variant="outline"
              size="icon-sm"
              aria-label="Model Settings"
              title="Model Settings"
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
            title="Filters"
            description="Narrow down models + endpoints"
            resetLabel="Reset Filters"
            onReset={onResetFilters}
          >
            {({ get, set }) => (
              <>
                <DataTableFilterField label="Public Model Name">
                  <SearchSelect
                    options={modelGroupOptions}
                    value={(get(MODEL_NAME_COLUMN_ID) as string) ?? ALL_MODEL_GROUPS_VALUE}
                    onValueChange={(value) =>
                      set(MODEL_NAME_COLUMN_ID, value === ALL_MODEL_GROUPS_VALUE ? undefined : value)
                    }
                    placeholder="Filter by Public Model Name"
                    emptyText="No models found"
                  />
                </DataTableFilterField>
                <DataTableFilterField label="Model Access Group">
                  <SearchSelect
                    options={accessGroupOptions}
                    value={(get(ACCESS_GROUPS_COLUMN_ID) as string) ?? ALL_MODEL_GROUPS_VALUE}
                    onValueChange={(value) =>
                      set(ACCESS_GROUPS_COLUMN_ID, value === ALL_MODEL_GROUPS_VALUE ? undefined : value)
                    }
                    placeholder="Filter by Model Access Group"
                    emptyText="No model access groups found"
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
