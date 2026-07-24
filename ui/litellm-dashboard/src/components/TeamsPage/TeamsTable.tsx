"use client";

import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeamsTable } from "@/app/(dashboard)/hooks/teams/useTeams";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
} from "@/components/shared/DataTable";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Input } from "@/components/ui/input";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import React, { useCallback, useMemo, useState } from "react";

import { Team } from "../key_team_helpers/key_list";
import { getTeamTableColumns, TEAM_TABLE_HIDDEN_COLUMNS } from "./teamTableColumns";

interface TeamsTableProps {
  userRole: string | null;
  userID: string | null;
  onSelectTeam: (team: Team) => void;
  onEditTeam: (team: Team) => void;
  onDeleteTeam: (team: Team) => void;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

const toSortOrder = (sorting: SortingState): "asc" | "desc" | undefined => {
  const active = sorting[0];
  if (!active) return undefined;
  return active.desc ? "desc" : "asc";
};

const FILTER_LABELS: Record<string, string> = {
  org_id: "Organization",
  alias: "Team alias",
  team_id: "Team ID",
};

export function TeamsTable({ userRole, userID, onSelectTeam, onEditTeam, onDeleteTeam }: TeamsTableProps) {
  const { data: fetchedOrganizations } = useOrganizations();
  const organizations = useMemo(() => fetchedOrganizations ?? [], [fetchedOrganizations]);

  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [tablePagination, setTablePagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 50 });
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });

  const getFilterValue = useCallback(
    (columnId: string): string | undefined => {
      const entry = columnFilters.find((filter) => filter.id === columnId);
      return typeof entry?.value === "string" && entry.value.trim() ? entry.value.trim() : undefined;
    },
    [columnFilters],
  );

  const isAdminView = userRole === "Admin" || userRole === "Admin Viewer";

  const teamListOptions = {
    organizationID: getFilterValue("org_id"),
    team_alias: getFilterValue("alias"),
    teamID: getFilterValue("team_id"),
    search: searchQuery.trim() || undefined,
    userID: isAdminView ? undefined : userID ?? undefined,
    sortBy: sorting[0]?.id,
    sortOrder: toSortOrder(sorting),
  };

  const {
    data: teamsResponse,
    isPending: isLoading,
    isFetching,
    refetch,
  } = useTeamsTable(tablePagination.pageIndex + 1, tablePagination.pageSize, teamListOptions);

  const teamList = useMemo<Team[]>(() => teamsResponse?.teams ?? [], [teamsResponse]);
  const rowCount = teamsResponse?.total ?? 0;

  const handleSearchChange = useCallback((value: string) => {
    setSearchInput(value);
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  const handleSortingChange = useCallback<OnChangeFn<SortingState>>((updaterOrValue) => {
    setSorting(updaterOrValue);
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  const handleColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>((updaterOrValue) => {
    setColumnFilters(updaterOrValue);
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

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
      defaultColumnVisibility={TEAM_TABLE_HIDDEN_COLUMNS}
      sortingMode="server"
      sorting={sorting}
      onSortingChange={handleSortingChange}
      paginationMode="server"
      pagination={tablePagination}
      onPaginationChange={setTablePagination}
      rowCount={rowCount}
      filterMode="server"
      columnFilters={columnFilters}
      onColumnFiltersChange={handleColumnFiltersChange}
      enableColumnResizing
      columnResizeMode="onChange"
      isLoading={isLoading}
      loadingMessage="Loading teams..."
      noDataMessage="No teams found"
      maxBodyHeight="calc(75vh - 210px)"
      size="compact"
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchInput}
            onSearchChange={handleSearchChange}
            searchPlaceholder="Search teams by name or ID…"
            onRefresh={() => refetch?.()}
            isRefreshing={isFetching}
            onOpenFilters={() => setFiltersOpen(true)}
            filterLabels={FILTER_LABELS}
            formatFilterValue={formatFilterValue}
          />
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
                    onValueChange={(value) => set("org_id", value)}
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
