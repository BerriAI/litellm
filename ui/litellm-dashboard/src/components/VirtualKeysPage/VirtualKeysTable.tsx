"use client";

import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useAllTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
} from "@/components/shared/DataTable";
import { PageHeader } from "@/components/shared/PageHeader";
import { Input } from "@/components/ui/input";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Select } from "antd";
import { KeyRound } from "lucide-react";
import React, { useCallback, useMemo, useState } from "react";

import { PaginatedKeyAliasSelect } from "../KeyAliasSelect/PaginatedKeyAliasSelect/PaginatedKeyAliasSelect";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import KeyInfoView from "../templates/key_info_view";
import { getKeyTableColumns, KEY_TABLE_HIDDEN_COLUMNS } from "./keyTableColumns";

interface VirtualKeysTableProps {
  headerActions?: React.ReactNode;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

const toSortOrder = (sorting: SortingState): "asc" | "desc" | undefined => {
  const active = sorting[0];
  if (!active) return undefined;
  return active.desc ? "desc" : "asc";
};

const FILTER_LABELS: Record<string, string> = {
  team_id: "Team ID",
  org_id: "Organization ID",
  key_alias: "Key Alias",
  user_id: "User ID",
  key_hash: "Key ID",
};

export function VirtualKeysTable({ headerActions }: VirtualKeysTableProps) {
  const { data: fetchedOrganizations } = useOrganizations();
  const organizations = useMemo(() => fetchedOrganizations ?? [], [fetchedOrganizations]);
  const { data: fetchedTeams } = useAllTeams();
  const allTeams = useMemo<Team[]>(() => fetchedTeams ?? [], [fetchedTeams]);

  const [selectedKey, setSelectedKey] = useState<KeyResponse | null>(null);
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [tablePagination, setTablePagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 50 });
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery] = useDebouncedValue(searchInput, { wait: 300 });

  const getFilterValue = useCallback(
    (columnId: string): string | undefined => {
      const entry = columnFilters.find((filter) => filter.id === columnId);
      return typeof entry?.value === "string" && entry.value.trim() ? entry.value.trim() : undefined;
    },
    [columnFilters],
  );

  const sortBy = sorting[0]?.id;
  const sortOrder = toSortOrder(sorting);

  const keyListOptions = {
    teamID: getFilterValue("team_id"),
    organizationID: getFilterValue("org_id"),
    selectedKeyAlias: searchQuery.trim() || getFilterValue("key_alias"),
    userID: getFilterValue("user_id"),
    keyHash: getFilterValue("key_hash"),
    sortBy,
    sortOrder,
    expand: "user",
  };

  const {
    data: keys,
    isPending: isLoading,
    isFetching,
    refetch,
  } = useKeys(tablePagination.pageIndex + 1, tablePagination.pageSize, keyListOptions);

  const keyList = useMemo(() => keys?.keys ?? [], [keys]);
  const rowCount = keys?.total_count ?? 0;

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

  const columns = useMemo(
    () => getKeyTableColumns({ allTeams, organizations, onSelectKey: setSelectedKey }),
    [allTeams, organizations],
  );

  const teamOptions = useMemo(
    () =>
      allTeams.map((team) => ({
        label: `${team.team_alias || team.team_id} (${team.team_id})`,
        value: team.team_id,
      })),
    [allTeams],
  );

  const orgOptions = useMemo(
    () =>
      organizations
        .filter((org) => org.organization_id)
        .map((org) => ({
          label: `${org.organization_alias || org.organization_id} (${org.organization_id})`,
          value: org.organization_id as string,
        })),
    [organizations],
  );

  if (selectedKey) {
    return (
      <div className="w-full h-full overflow-hidden">
        <KeyInfoView
          keyId={selectedKey.token}
          onClose={() => setSelectedKey(null)}
          keyData={selectedKey}
          teams={allTeams}
          onDelete={refetch}
        />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-hidden py-2">
      <PageHeader
        icon={<KeyRound className="size-5" />}
        title="Virtual Keys"
        subtitle="Every key that authenticates requests to the gateway."
        actions={headerActions}
      />
      <DataTable
        data={keyList}
        columns={columns}
        getRowId={(row) => row.token}
        defaultColumnVisibility={KEY_TABLE_HIDDEN_COLUMNS}
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
        onRowClick={setSelectedKey}
        enableColumnResizing
        columnResizeMode="onChange"
        isLoading={isLoading}
        loadingMessage="Loading keys..."
        noDataMessage="No keys found"
        maxBodyHeight="calc(75vh - 210px)"
        size="compact"
        toolbar={(table) => (
          <>
            <DataTableToolbar
              table={table}
              searchValue={searchInput}
              onSearchChange={handleSearchChange}
              searchPlaceholder="Search keys, aliases, users…"
              onRefresh={() => refetch?.()}
              isRefreshing={isFetching}
              onOpenFilters={() => setFiltersOpen(true)}
              filterLabels={FILTER_LABELS}
            />
            <DataTableFilterDrawer
              table={table}
              open={filtersOpen}
              onOpenChange={setFiltersOpen}
              title="Filters"
              description="Narrow down virtual keys"
            >
              {({ get, set }) => (
                <>
                  <DataTableFilterField label="Team ID">
                    <Select
                      showSearch
                      allowClear
                      className="w-full"
                      placeholder="Search Team ID…"
                      optionFilterProp="label"
                      value={(get("team_id") as string) || undefined}
                      onChange={(value) => set("team_id", value ?? "")}
                      options={teamOptions}
                    />
                  </DataTableFilterField>
                  <DataTableFilterField label="Organization ID">
                    <Select
                      showSearch
                      allowClear
                      className="w-full"
                      placeholder="Search Organization ID…"
                      optionFilterProp="label"
                      value={(get("org_id") as string) || undefined}
                      onChange={(value) => set("org_id", value ?? "")}
                      options={orgOptions}
                    />
                  </DataTableFilterField>
                  <DataTableFilterField label="Key Alias">
                    <PaginatedKeyAliasSelect
                      value={(get("key_alias") as string) || undefined}
                      onChange={(value) => set("key_alias", value ?? "")}
                      placeholder="Select Key Alias…"
                    />
                  </DataTableFilterField>
                  <DataTableFilterField label="User ID">
                    <Input
                      value={(get("user_id") as string) ?? ""}
                      onChange={(event) => set("user_id", event.target.value)}
                      placeholder="Enter User ID…"
                    />
                  </DataTableFilterField>
                  <DataTableFilterField label="Key ID">
                    <Input
                      value={(get("key_hash") as string) ?? ""}
                      onChange={(event) => set("key_hash", event.target.value)}
                      placeholder="Enter Key ID…"
                    />
                  </DataTableFilterField>
                </>
              )}
            </DataTableFilterDrawer>
          </>
        )}
      />
    </div>
  );
}
