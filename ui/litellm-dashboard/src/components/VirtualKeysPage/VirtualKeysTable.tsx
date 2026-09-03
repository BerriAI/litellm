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
} from "@/components/shared/DataTable";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { PageHeader } from "@/components/shared/PageHeader";
import { Input } from "@/components/ui/input";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnFiltersState, functionalUpdate, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { KeyRound } from "lucide-react";
import { createParser, parseAsInteger, parseAsString, parseAsStringLiteral, useQueryState, useQueryStates } from "nuqs";
import React, { useCallback, useMemo, useState } from "react";

import { KeyResponse, Team } from "../key_team_helpers/key_list";
import KeyInfoView from "../templates/key_info_view";
import { getKeyTableColumns, KEY_TABLE_HIDDEN_COLUMNS, KEY_TABLE_SORT_FIELDS } from "./keyTableColumns";

interface VirtualKeysTableProps {
  headerActions?: React.ReactNode;
}

const FILTER_COLUMNS = ["team_id", "org_id", "user_id", "key_hash"] as const;
type FilterColumn = (typeof FILTER_COLUMNS)[number];

const FILTER_LABELS: Record<FilterColumn, string> = {
  team_id: "Team",
  org_id: "Organization",
  user_id: "User ID",
  key_hash: "Key ID",
};

const DEFAULT_SORT_BY = "created_at";
const DEFAULT_SORT_ORDER = "desc";
const DEFAULT_PAGE_SIZE = 50;
const MAX_PAGE_SIZE = 100;
const MAX_PAGE = 100_000;

const boundedInteger = (min: number, max: number, fallback: number) =>
  createParser({
    parse: (value: string) => {
      const parsed = parseAsInteger.parse(value);
      return parsed === null ? null : Math.min(Math.max(parsed, min), max);
    },
    serialize: String,
  }).withDefault(fallback);

// The filters carry a prefix because /api-keys also takes team_id, key_alias and key_type
// as create-key prefills; an unprefixed filter would hijack those deep links.
const TABLE_STATE = {
  key_search: parseAsString.withDefault(""),
  sort_by: parseAsString.withDefault(DEFAULT_SORT_BY),
  sort_order: parseAsStringLiteral(["asc", "desc"] as const).withDefault(DEFAULT_SORT_ORDER),
  page: boundedInteger(1, MAX_PAGE, 1),
  page_size: boundedInteger(1, MAX_PAGE_SIZE, DEFAULT_PAGE_SIZE),
  filter_team: parseAsString.withDefault(""),
  filter_org: parseAsString.withDefault(""),
  filter_user: parseAsString.withDefault(""),
  filter_key_id: parseAsString.withDefault(""),
};

const toSortOrder = (active: SortingState[number]): "asc" | "desc" => (active.desc ? "desc" : "asc");

const filterValue = (filters: ColumnFiltersState, column: FilterColumn): string | null => {
  const value = filters.find((filter) => filter.id === column)?.value;
  return (typeof value === "string" ? value.trim() : "") || null;
};

export function VirtualKeysTable({ headerActions }: VirtualKeysTableProps) {
  const { data: fetchedOrganizations } = useOrganizations();
  const organizations = useMemo(() => fetchedOrganizations ?? [], [fetchedOrganizations]);
  const { data: fetchedTeams } = useAllTeams();
  const allTeams = useMemo<Team[]>(() => fetchedTeams ?? [], [fetchedTeams]);

  const [selectedKeyId, setSelectedKeyId] = useQueryState("key", parseAsString.withOptions({ history: "push" }));
  const [tableState, setTableState] = useQueryStates(TABLE_STATE);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const searchInput = tableState.key_search;
  const [searchQuery] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });

  // A hand-edited sort_by the table cannot sort by would 400 at /key/list and leave the page loading.
  const sortBy = KEY_TABLE_SORT_FIELDS.includes(tableState.sort_by) ? tableState.sort_by : DEFAULT_SORT_BY;
  const sorting = useMemo<SortingState>(
    () => [{ id: sortBy, desc: tableState.sort_order === "desc" }],
    [sortBy, tableState.sort_order],
  );
  const tablePagination = useMemo<PaginationState>(
    () => ({ pageIndex: tableState.page - 1, pageSize: tableState.page_size }),
    [tableState.page, tableState.page_size],
  );
  const { filter_team, filter_org, filter_user, filter_key_id } = tableState;
  const appliedFilters = useMemo(
    () => ({
      team_id: filter_team.trim(),
      org_id: filter_org.trim(),
      user_id: filter_user.trim(),
      key_hash: filter_key_id.trim(),
    }),
    [filter_team, filter_org, filter_user, filter_key_id],
  );
  const columnFilters = useMemo<ColumnFiltersState>(
    () =>
      FILTER_COLUMNS.filter((column) => appliedFilters[column]).map((column) => ({
        id: column,
        value: appliedFilters[column],
      })),
    [appliedFilters],
  );

  const keyListOptions = {
    teamID: appliedFilters.team_id || undefined,
    organizationID: appliedFilters.org_id || undefined,
    selectedKeyAlias: searchQuery.trim() || undefined,
    userID: appliedFilters.user_id || undefined,
    keyHash: appliedFilters.key_hash || undefined,
    sortBy,
    sortOrder: tableState.sort_order,
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

  const handleSearchChange = useCallback(
    (value: string) => {
      void setTableState({ key_search: value || null, page: null });
    },
    [setTableState],
  );

  const handleSortingChange = useCallback<OnChangeFn<SortingState>>(
    (updaterOrValue) => {
      const active = functionalUpdate(updaterOrValue, sorting)[0];
      void setTableState({
        sort_by: active?.id ?? null,
        sort_order: active ? toSortOrder(active) : null,
        page: null,
      });
    },
    [sorting, setTableState],
  );

  const handleColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>(
    (updaterOrValue) => {
      const next = functionalUpdate(updaterOrValue, columnFilters);
      const nextFilters = {
        filter_team: filterValue(next, "team_id"),
        filter_org: filterValue(next, "org_id"),
        filter_user: filterValue(next, "user_id"),
        filter_key_id: filterValue(next, "key_hash"),
        page: null,
      };
      void setTableState(nextFilters);
    },
    [columnFilters, setTableState],
  );

  const handlePaginationChange = useCallback<OnChangeFn<PaginationState>>(
    (updaterOrValue) => {
      const next = functionalUpdate(updaterOrValue, tablePagination);
      void setTableState({ page: next.pageIndex + 1, page_size: next.pageSize });
    },
    [tablePagination, setTableState],
  );

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
    <div className="flex h-full flex-col gap-6 overflow-hidden">
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
        defaultColumnVisibility={KEY_TABLE_HIDDEN_COLUMNS}
        sortingMode="server"
        sorting={sorting}
        onSortingChange={handleSortingChange}
        paginationMode="server"
        pagination={tablePagination}
        onPaginationChange={handlePaginationChange}
        rowCount={rowCount}
        filterMode="server"
        columnFilters={columnFilters}
        onColumnFiltersChange={handleColumnFiltersChange}
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
              searchPlaceholder="Search by key alias…"
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
                      onValueChange={(value) => set("team_id", value)}
                      placeholder="Select a team…"
                      emptyText="No teams found"
                    />
                  </DataTableFilterField>
                  <DataTableFilterField label="Organization">
                    <SearchSelect
                      options={orgOptions}
                      value={(get("org_id") as string) || undefined}
                      onValueChange={(value) => set("org_id", value)}
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
                </>
              )}
            </DataTableFilterDrawer>
          </>
        )}
      />
    </div>
  );
}
