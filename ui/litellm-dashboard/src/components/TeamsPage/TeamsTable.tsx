"use client";

import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeamsTable } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
  usePersistedColumnVisibility,
  useUrlTableState,
  type UrlTableStateOptions,
} from "@/components/shared/DataTable";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnFiltersState } from "@tanstack/react-table";
import { Download } from "lucide-react";
import React, { useCallback, useMemo, useState } from "react";

import { Team } from "../key_team_helpers/key_list";
import { getTeamTableColumns, TEAM_TABLE_HIDDEN_COLUMNS } from "./teamTableColumns";
import { exportTeamsToCsv } from "./teamsCsvExport";

interface TeamsTableProps {
  userRole: string | null;
  userID: string | null;
  onSelectTeam: (team: Team) => void;
  onEditTeam: (team: Team) => void;
  onDeleteTeam: (team: Team) => void;
}

const FILTER_COLUMNS = ["org_id", "alias", "team_id"] as const;
type FilterColumn = (typeof FILTER_COLUMNS)[number];

const FILTER_LABELS: Record<FilterColumn, string> = {
  org_id: "Organization",
  alias: "Team alias",
  team_id: "Team ID",
};

const TABLE_STATE_OPTIONS: UrlTableStateOptions<FilterColumn> = {
  sortFields: ["team_alias", "created_at"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 50,
  maxPageSize: 100,
  filterColumns: FILTER_COLUMNS,
  urlKeys: { search: "team_search", filter_org_id: "filter_org" },
};

const appliedFilter = (filters: ColumnFiltersState, column: FilterColumn): string | undefined => {
  const value = filters.find((filter) => filter.id === column)?.value;
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
};

export function TeamsTable({ userRole, userID, onSelectTeam, onEditTeam, onDeleteTeam }: TeamsTableProps) {
  const { data: fetchedOrganizations } = useOrganizations();
  const organizations = useMemo(() => fetchedOrganizations ?? [], [fetchedOrganizations]);

  const {
    search: searchInput,
    setSearch,
    sorting,
    onSortingChange,
    pagination,
    onPaginationChange,
    columnFilters,
    onColumnFiltersChange,
  } = useUrlTableState(TABLE_STATE_OPTIONS);
  const { columnVisibility, onColumnVisibilityChange } = usePersistedColumnVisibility(
    "teams",
    TEAM_TABLE_HIDDEN_COLUMNS,
  );
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [searchQuery] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });
  const { accessToken } = useAuthorized();

  const isAdminView = userRole === "Admin" || userRole === "Admin Viewer";

  const [activeSort] = sorting;
  const teamListOptions = useMemo(
    () => ({
      organizationID: appliedFilter(columnFilters, "org_id"),
      team_alias: appliedFilter(columnFilters, "alias"),
      teamID: appliedFilter(columnFilters, "team_id"),
      search: searchQuery.trim() || undefined,
      searchTeamIdMatch: "prefix" as const,
      userID: isAdminView ? undefined : userID ?? undefined,
      sortBy: activeSort.id,
      sortOrder: activeSort.desc ? "desc" : "asc",
    }),
    [columnFilters, searchQuery, isAdminView, userID, activeSort],
  );

  const {
    data: teamsResponse,
    isPending,
    isPlaceholderData,
    isFetching,
    isError,
    refetch,
  } = useTeamsTable(pagination.pageIndex + 1, pagination.pageSize, teamListOptions);

  const teamList = useMemo<Team[]>(() => teamsResponse?.teams ?? [], [teamsResponse]);
  const rowCount = teamsResponse?.total ?? 0;

  const handleExportCsv = useCallback(async () => {
    if (!accessToken || isExporting) return;
    setIsExporting(true);
    try {
      await exportTeamsToCsv(accessToken, teamListOptions);
    } finally {
      setIsExporting(false);
    }
  }, [accessToken, isExporting, teamListOptions]);

  const columns = useMemo(() => {
    const columnDeps = { organizations, userRole, onSelectTeam, onEditTeam, onDeleteTeam };
    return getTeamTableColumns(columnDeps);
  }, [organizations, userRole, onSelectTeam, onEditTeam, onDeleteTeam]);

  const orgOptions = useMemo(
    () =>
      organizations
        .filter((org) => org.organization_id)
        .map((org) => {
          const id = org.organization_id as string;
          return { label: org.organization_alias || id, value: id, sublabel: org.organization_alias ? id : undefined };
        }),
    [organizations],
  );

  const formatFilterValue = useCallback(
    (columnId: string, value: unknown): string => {
      const raw = String(value);
      if (columnId === "org_id") {
        return organizations.find((org) => org.organization_id === raw)?.organization_alias || raw;
      }
      return raw;
    },
    [organizations],
  );

  return (
    <DataTable
      data={teamList}
      columns={columns}
      getRowId={(row) => row.team_id}
      columnVisibility={columnVisibility}
      onColumnVisibilityChange={onColumnVisibilityChange}
      sortingMode="server"
      sorting={sorting}
      onSortingChange={onSortingChange}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={rowCount}
      isError={isError}
      filterMode="server"
      columnFilters={columnFilters}
      onColumnFiltersChange={onColumnFiltersChange}
      enableColumnResizing
      columnResizeMode="onChange"
      isLoading={isPending || isPlaceholderData}
      loadingMessage="Loading teams..."
      noDataMessage="No teams found"
      fillHeight
      size="compact"
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchInput}
            onSearchChange={setSearch}
            searchPlaceholder="Search teams by name or ID…"
            onRefresh={() => refetch?.()}
            isRefreshing={isFetching}
            onOpenFilters={() => setFiltersOpen(true)}
            filterLabels={FILTER_LABELS}
            formatFilterValue={formatFilterValue}
          >
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportCsv}
              disabled={isExporting}
              data-testid="teams-export-csv"
            >
              <Download />
              {isExporting ? "Exporting..." : "Export CSV"}
            </Button>
          </DataTableToolbar>
          <DataTableFilterDrawer
            table={table}
            open={filtersOpen}
            onOpenChange={setFiltersOpen}
            title="Filters"
            description="Narrow down your teams"
          >
            {({ get, set }) => (
              <>
                <DataTableFilterField label="Organization">
                  <SearchSelect
                    options={orgOptions}
                    value={(get("org_id") as string) || undefined}
                    onValueChange={(value) => set("org_id", value ?? undefined)}
                    placeholder="Select an organization…"
                    emptyText="No organizations found"
                  />
                </DataTableFilterField>
                <DataTableFilterField label="Team alias">
                  <Input
                    value={(get("alias") as string) ?? ""}
                    onChange={(event) => set("alias", event.target.value)}
                    placeholder="Enter team alias…"
                  />
                </DataTableFilterField>
                <DataTableFilterField label="Team ID">
                  <Input
                    value={(get("team_id") as string) ?? ""}
                    onChange={(event) => set("team_id", event.target.value)}
                    placeholder="Enter team ID…"
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
