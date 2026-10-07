"use client";

import { useKeyInfo } from "@/app/(dashboard)/hooks/keys/useKeyInfo";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useAllTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
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
import { PageHeader } from "@/components/shared/PageHeader";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnFiltersState, functionalUpdate, OnChangeFn } from "@tanstack/react-table";
import { KeyRound } from "lucide-react";
import { parseAsString, useQueryState } from "nuqs";
import React, { useCallback, useMemo, useState } from "react";

import { KeyResponse, Team } from "../key_team_helpers/key_list";
import KeyInfoView from "../templates/key_info_view";
import { getKeyTableColumns, KEY_TABLE_HIDDEN_COLUMNS, KEY_TABLE_SORT_FIELDS } from "./keyTableColumns";

interface VirtualKeysTableProps {
  headerActions?: React.ReactNode;
}

const FILTER_COLUMNS = ["team_id", "org_id", "user_id", "key_hash", "status"] as const;
type FilterColumn = (typeof FILTER_COLUMNS)[number];

const FILTER_LABELS: Record<FilterColumn, string> = {
  team_id: "Team",
  org_id: "Organization",
  user_id: "User ID",
  key_hash: "Key ID",
  status: "Status",
};

const KEY_STATUS_VALUES = ["active", "expired", "revoked", "deleted"] as const;
type KeyStatusFilter = (typeof KEY_STATUS_VALUES)[number];
const ALL_STATUSES = "all";

const KEY_STATUS_LABELS: Record<KeyStatusFilter, string> = {
  active: "Active",
  expired: "Expired",
  revoked: "Revoked (blocked)",
  deleted: "Deleted",
};

const STATUS_FILTER_ITEMS = [
  { value: ALL_STATUSES, label: "All statuses" },
  ...KEY_STATUS_VALUES.map((value) => ({ value, label: KEY_STATUS_LABELS[value] })),
];

const isKeyStatusFilter = (value: unknown): value is KeyStatusFilter =>
  (KEY_STATUS_VALUES as readonly unknown[]).includes(value);

const isUsableFilter = (filter: ColumnFiltersState[number]): boolean =>
  filter.id !== "status" || isKeyStatusFilter(filter.value);

const TABLE_STATE_OPTIONS: UrlTableStateOptions<FilterColumn> = {
  sortFields: KEY_TABLE_SORT_FIELDS,
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 50,
  maxPageSize: 100,
  filterColumns: FILTER_COLUMNS,
  urlKeys: {
    search: "key_search",
    filter_team_id: "filter_team",
    filter_org_id: "filter_org",
    filter_user_id: "filter_user",
    filter_key_hash: "filter_key_id",
  },
};

const appliedFilter = (filters: ColumnFiltersState, column: FilterColumn): string | undefined => {
  const value = filters.find((filter) => filter.id === column)?.value;
  return typeof value === "string" ? value : undefined;
};

export function VirtualKeysTable({ headerActions }: VirtualKeysTableProps) {
  const { data: fetchedOrganizations } = useOrganizations();
  const organizations = useMemo(() => fetchedOrganizations ?? [], [fetchedOrganizations]);
  const { data: fetchedTeams } = useAllTeams();
  const allTeams = useMemo<Team[]>(() => fetchedTeams ?? [], [fetchedTeams]);

  const [selectedKeyId, setSelectedKeyId] = useQueryState("key", parseAsString.withOptions({ history: "push" }));
  const {
    search: searchInput,
    setSearch,
    sorting,
    onSortingChange,
    pagination,
    onPaginationChange,
    columnFilters: urlColumnFilters,
    onColumnFiltersChange: setUrlColumnFilters,
  } = useUrlTableState(TABLE_STATE_OPTIONS);
  const columnFilters = useMemo(() => urlColumnFilters.filter(isUsableFilter), [urlColumnFilters]);
  const onColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>(
    (updaterOrValue) => setUrlColumnFilters(functionalUpdate(updaterOrValue, columnFilters)),
    [columnFilters, setUrlColumnFilters],
  );
  const { columnVisibility, onColumnVisibilityChange } = usePersistedColumnVisibility(
    "virtual-keys",
    KEY_TABLE_HIDDEN_COLUMNS,
  );
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [searchQuery] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });

  const [activeSort] = sorting;
  const keyListOptions = {
    teamID: appliedFilter(columnFilters, "team_id"),
    organizationID: appliedFilter(columnFilters, "org_id"),
    search: searchQuery.trim() || undefined,
    userID: appliedFilter(columnFilters, "user_id"),
    keyHash: appliedFilter(columnFilters, "key_hash"),
    status: appliedFilter(columnFilters, "status"),
    sortBy: activeSort.id,
    sortOrder: activeSort.desc ? "desc" : "asc",
    expand: "user",
  };

  const {
    data: keys,
    isPending,
    isPlaceholderData,
    isFetching,
    isError,
    refetch,
  } = useKeys(pagination.pageIndex + 1, pagination.pageSize, keyListOptions);

  const keyList = useMemo(() => keys?.keys ?? [], [keys]);
  const rowCount = keys?.total_count ?? 0;

  const columns = useMemo(
    () => getKeyTableColumns({ allTeams, organizations, onSelectKey: (key) => void setSelectedKeyId(key.token) }),
    [allTeams, organizations, setSelectedKeyId],
  );

  const selectedKeyFromList = useMemo(
    () => keyList.find((key) => key.token === selectedKeyId),
    [keyList, selectedKeyId],
  );
  const { data: fetchedSelectedKey, isError: selectedKeyLoadFailed } = useKeyInfo(selectedKeyId, {
    enabled: !selectedKeyFromList,
  });
  const selectedKey = selectedKeyFromList ?? fetchedSelectedKey;

  const teamOptions = useMemo(
    () =>
      allTeams.map((team) => ({
        label: team.team_alias || team.team_id,
        value: team.team_id,
        sublabel: team.team_alias ? team.team_id : undefined,
      })),
    [allTeams],
  );

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

  const handleSelectedKeyDataUpdate = useCallback(
    (updated: Partial<KeyResponse>) => {
      const rotatedToken = updated.token ?? updated.token_id;
      if (!rotatedToken || rotatedToken === selectedKeyId) return;
      void setSelectedKeyId(rotatedToken, { history: "replace" });
      void refetch();
    },
    [refetch, selectedKeyId, setSelectedKeyId],
  );

  const formatFilterValue = useCallback(
    (columnId: string, value: unknown): string => {
      const raw = String(value);
      if (columnId === "team_id") {
        return allTeams.find((team) => team.team_id === raw)?.team_alias || raw;
      }
      if (columnId === "org_id") {
        return organizations.find((org) => org.organization_id === raw)?.organization_alias || raw;
      }
      if (columnId === "status" && isKeyStatusFilter(raw)) {
        return KEY_STATUS_LABELS[raw];
      }
      return raw;
    },
    [allTeams, organizations],
  );

  if (selectedKeyId) {
    if (!selectedKey && !selectedKeyLoadFailed) {
      return <div className="p-4 text-sm text-muted-foreground">Loading key...</div>;
    }
    return (
      <div className="w-full h-full overflow-hidden">
        <KeyInfoView
          keyId={selectedKeyId}
          onClose={() => void setSelectedKeyId(null)}
          keyData={selectedKey}
          teams={allTeams}
          onDelete={refetch}
          onKeyDataUpdate={handleSelectedKeyDataUpdate}
        />
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6">
      <PageHeader
        icon={<KeyRound />}
        title="Virtual Keys"
        subtitle="Every key that authenticates requests to the gateway."
        primaryAction={headerActions}
      />
      <DataTable
        data={keyList}
        columns={columns}
        getRowId={(row) => row.token}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={onColumnVisibilityChange}
        sortingMode="server"
        sorting={sorting}
        onSortingChange={onSortingChange}
        paginationMode="server"
        pagination={pagination}
        onPaginationChange={onPaginationChange}
        rowCount={rowCount}
        filterMode="server"
        columnFilters={columnFilters}
        onColumnFiltersChange={onColumnFiltersChange}
        enableColumnResizing
        columnResizeMode="onChange"
        isLoading={isPending || isPlaceholderData}
        isError={isError}
        loadingMessage="Loading keys..."
        noDataMessage="No keys found"
        fillHeight
        size="compact"
        toolbar={(table) => (
          <>
            <DataTableToolbar
              table={table}
              searchValue={searchInput}
              onSearchChange={setSearch}
              searchPlaceholder="Search by key alias or ID…"
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
              description="Narrow down virtual keys"
            >
              {({ get, set }) => (
                <>
                  <DataTableFilterField label="Team">
                    <SearchSelect
                      options={teamOptions}
                      value={(get("team_id") as string) || undefined}
                      onValueChange={(value) => set("team_id", value ?? undefined)}
                      placeholder="Select a team…"
                      emptyText="No teams found"
                    />
                  </DataTableFilterField>
                  <DataTableFilterField label="Organization">
                    <SearchSelect
                      options={orgOptions}
                      value={(get("org_id") as string) || undefined}
                      onValueChange={(value) => set("org_id", value ?? undefined)}
                      placeholder="Select an organization…"
                      emptyText="No organizations found"
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
                  <DataTableFilterField label="Status">
                    <Select
                      items={STATUS_FILTER_ITEMS}
                      value={(get("status") as string) || ALL_STATUSES}
                      onValueChange={(value) => set("status", value === ALL_STATUSES ? undefined : value)}
                    >
                      <SelectTrigger className="w-full" aria-label="Status">
                        <SelectValue placeholder="All statuses" />
                      </SelectTrigger>
                      <SelectContent>
                        {STATUS_FILTER_ITEMS.map((item) => (
                          <SelectItem key={item.value} value={item.value}>
                            {item.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
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
