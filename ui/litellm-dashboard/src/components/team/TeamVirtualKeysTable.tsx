"use client";
import { useKeyInfo } from "@/app/(dashboard)/hooks/keys/useKeyInfo";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { SimpleTooltip } from "@/components/ui/tooltip";
import {
  DateCell,
  ENTITY_CELL_TITLE_CLASSES,
  IdCell,
  IdentityCell,
  MoneyCell,
  UserPopoverCell,
} from "@/components/shared/table_cells";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableSortHeader,
  DataTableToolbar,
  useUrlTableState,
  type UrlTableStateOptions,
} from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { orgDetailHref, userDetailHref } from "@/utils/entityLinks";
import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnDef, ColumnFiltersState } from "@tanstack/react-table";
import { ChevronDown, ChevronRight } from "lucide-react";
import { parseAsString, useQueryState } from "nuqs";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { deriveKeyModelScope } from "../key_scope";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import { Organization } from "../networking";
import KeyInfoView from "../templates/key_info_view";
import { SELECTED_TEAM_KEY_URL_KEY, TEAM_KEYS_FILTER_URL_KEYS, TEAM_KEYS_URL_PREFIX } from "./useTeamDetailUrlState";

interface TeamVirtualKeysTableProps {
  teamId: string;
  teamAlias?: string;
  organization: Organization | null;
}

const FILTER_COLUMNS = ["user_id", "key_hash"] as const;
type FilterColumn = (typeof FILTER_COLUMNS)[number];

const SORT_FIELDS = ["token", "key_alias", "created_at", "updated_at", "spend", "max_budget"] as const;

const TABLE_STATE_OPTIONS: UrlTableStateOptions<FilterColumn> = {
  sortFields: SORT_FIELDS,
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 50,
  maxPageSize: 100,
  filterColumns: FILTER_COLUMNS,
  keyPrefix: TEAM_KEYS_URL_PREFIX,
  urlKeys: TEAM_KEYS_FILTER_URL_KEYS,
};

const appliedFilter = (filters: ColumnFiltersState, column: FilterColumn): string | undefined => {
  const value = filters.find((filter) => filter.id === column)?.value;
  return typeof value === "string" ? value : undefined;
};

export function TeamVirtualKeysTable({ teamId, teamAlias, organization }: TeamVirtualKeysTableProps) {
  const [selectedKeyId, setSelectedKeyId] = useQueryState(
    SELECTED_TEAM_KEY_URL_KEY,
    parseAsString.withOptions({ history: "push" }),
  );
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
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [searchQuery] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });

  const [activeSort] = sorting;
  const keyListOptions = {
    teamID: teamId,
    search: searchQuery.trim() || undefined,
    userID: appliedFilter(columnFilters, "user_id"),
    keyHash: appliedFilter(columnFilters, "key_hash"),
    sortBy: activeSort.id,
    sortOrder: activeSort.desc ? "desc" : "asc",
    expand: "user",
  };

  const {
    data: keys,
    isPending: isLoading,
    isFetching,
    isError,
    refetch,
  } = useKeys(pagination.pageIndex + 1, pagination.pageSize, keyListOptions);

  const displayKeys = useMemo(() => {
    const kList = keys?.keys || [];
    const orgId = organization?.organization_id;
    if (!orgId) return kList;
    return kList.map((k: KeyResponse) => ({
      ...k,
      organization_id: (k.organization_id ?? k.org_id) || orgId,
    }));
  }, [keys?.keys, organization?.organization_id]);

  const rowCount = keys?.total_count ?? 0;

  const selectedKeyFromList = useMemo(
    () => displayKeys.find((key: KeyResponse) => key.token === selectedKeyId),
    [displayKeys, selectedKeyId],
  );
  const { data: fetchedSelectedKey, isError: selectedKeyLoadFailed } = useKeyInfo(selectedKeyId, {
    enabled: !selectedKeyFromList,
  });
  const selectedKey = selectedKeyFromList ?? fetchedSelectedKey;

  const handleSelectedKeyDataUpdate = useCallback(
    (updated: Partial<KeyResponse>) => {
      const rotatedToken = updated.token ?? updated.token_id;
      if (!rotatedToken || rotatedToken === selectedKeyId) return;
      void setSelectedKeyId(rotatedToken, { history: "replace" });
      void refetch();
    },
    [refetch, selectedKeyId, setSelectedKeyId],
  );

  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});

  const currentTeam: Team = useMemo(
    () => ({
      team_id: teamId,
      team_alias: teamAlias || teamId,
      models: [],
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      organization_id: organization?.organization_id || "",
      created_at: "",
      keys: [],
      members_with_roles: [],
      spend: 0,
    }),
    [teamId, teamAlias, organization],
  );

  const handleStorageChange = useCallback(() => {
    refetch?.();
  }, [refetch]);

  useEffect(() => {
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, [handleStorageChange]);

  const columns: ColumnDef<KeyResponse>[] = useMemo(
    () => [
      {
        id: "token",
        accessorKey: "token",
        meta: { title: "Key ID" },
        header: ({ column }) => <DataTableSortHeader column={column} title="Key ID" variant="header-cycle" />,
        size: 120,
        enableSorting: true,
        cell: (info) => (
          <IdCell
            value={info.getValue() as string | null}
            onClick={() => void setSelectedKeyId(info.row.original.token)}
          />
        ),
      },
      {
        id: "key_alias",
        accessorKey: "key_alias",
        meta: { title: "Key Alias" },
        header: ({ column }) => <DataTableSortHeader column={column} title="Key Alias" variant="header-cycle" />,
        size: 150,
        enableSorting: true,
        cell: (info) => {
          const value = info.getValue() as string;
          return (
            <SimpleTooltip content={value}>
              <span className="block max-w-full truncate font-mono text-xs">{value ?? "-"}</span>
            </SimpleTooltip>
          );
        },
      },
      {
        id: "key_name",
        accessorKey: "key_name",
        header: "Secret Key",
        size: 120,
        enableSorting: false,
        cell: (info) => <span className="font-mono text-xs">{info.getValue() as string}</span>,
      },
      {
        id: "organization_id",
        accessorKey: "organization_id",
        header: "Organization ID",
        size: 140,
        enableSorting: false,
        cell: (info) => {
          const orgId = info.getValue() as string | null;
          if (!orgId) return "-";
          return (
            <SimpleTooltip content={orgId}>
              <IdentityCell title={orgId} titleClassName={ENTITY_CELL_TITLE_CLASSES} href={orgDetailHref(orgId)} />
            </SimpleTooltip>
          );
        },
      },
      {
        id: "user_email",
        accessorKey: "user",
        header: "User Email",
        size: 160,
        enableSorting: false,
        cell: (info) => {
          const user = info.getValue() as { user_email?: string } | undefined;
          const value = user?.user_email;
          const userId = info.row.original.user_id;
          return (
            <SimpleTooltip content={value}>
              <IdentityCell
                title={value ?? "-"}
                titleClassName={ENTITY_CELL_TITLE_CLASSES}
                href={value && userId ? userDetailHref(userId) : undefined}
              />
            </SimpleTooltip>
          );
        },
      },
      {
        id: "user_id",
        accessorKey: "user_id",
        header: "User ID",
        size: 70,
        enableSorting: false,
        cell: (info) => {
          const userId = info.getValue() as string | null;
          if (userId === DEFAULT_PROXY_ADMIN_USER_ID) {
            return <DefaultProxyAdminTag userId={userId} />;
          }
          return (
            <SimpleTooltip content={userId}>
              <IdentityCell
                title={userId ?? "-"}
                titleClassName={ENTITY_CELL_TITLE_CLASSES}
                href={userId ? userDetailHref(userId) : undefined}
              />
            </SimpleTooltip>
          );
        },
      },
      {
        id: "created_at",
        accessorKey: "created_at",
        meta: { title: "Created At" },
        header: ({ column }) => <DataTableSortHeader column={column} title="Created At" variant="header-cycle" />,
        size: 120,
        enableSorting: true,
        cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" />,
      },
      {
        id: "created_by",
        accessorKey: "created_by",
        header: "Created By",
        size: 130,
        enableSorting: false,
        cell: (info) => {
          const userId = info.getValue() as string | null;
          if (!userId) return "-";
          const { created_by_user } = info.row.original;
          return (
            <UserPopoverCell
              userAlias={created_by_user?.user_alias ?? null}
              userEmail={created_by_user?.user_email ?? null}
              userId={userId}
              width={130}
            />
          );
        },
      },
      {
        id: "updated_at",
        accessorKey: "updated_at",
        meta: { title: "Updated At" },
        header: ({ column }) => <DataTableSortHeader column={column} title="Updated At" variant="header-cycle" />,
        size: 120,
        enableSorting: true,
        cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" fallback="Never" />,
      },
      {
        id: "last_active",
        accessorKey: "last_active",
        header: "Last Active",
        size: 130,
        enableSorting: false,
        cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" fallback="Unknown" />,
      },
      {
        id: "expires",
        accessorKey: "expires",
        header: "Expires",
        size: 120,
        enableSorting: false,
        cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" fallback="Never" />,
      },
      {
        id: "spend",
        accessorKey: "spend",
        meta: { title: "Spend (USD)" },
        header: ({ column }) => <DataTableSortHeader column={column} title="Spend (USD)" variant="header-cycle" />,
        size: 100,
        enableSorting: true,
        cell: (info) => <MoneyCell value={info.getValue() as number | null} decimals={4} />,
      },
      {
        id: "max_budget",
        accessorKey: "max_budget",
        meta: { title: "Budget (USD)" },
        header: ({ column }) => <DataTableSortHeader column={column} title="Budget (USD)" variant="header-cycle" />,
        size: 110,
        enableSorting: true,
        cell: (info) => (
          <MoneyCell value={info.getValue() as number | null} decimals={0} emptyText="Unlimited" showZero />
        ),
      },
      {
        id: "budget_reset_at",
        accessorKey: "budget_reset_at",
        header: "Budget Reset",
        size: 130,
        enableSorting: false,
        cell: (info) => <DateCell value={info.getValue() as string | null} fallback="Never" />,
      },
      {
        id: "models",
        accessorKey: "models",
        header: "Models",
        size: 200,
        enableSorting: false,
        cell: (info) => {
          const models = info.getValue() as string[];
          const scope = deriveKeyModelScope(info.row.original.allowed_routes, info.row.original.key_type);
          const emptyModelsBadge = !scope.hasModelAccess ? (
            <SimpleTooltip content={`Scoped to ${scope.label} routes; this key cannot call any models`}>
              <Badge variant="secondary" className="mb-1">
                No model access
              </Badge>
            </SimpleTooltip>
          ) : (
            <Badge variant="destructive" className="mb-1">
              All Proxy Models
            </Badge>
          );
          return (
            <div className="flex flex-col py-2">
              {Array.isArray(models) ? (
                <div className="flex flex-col">
                  {models.length === 0 ? (
                    emptyModelsBadge
                  ) : (
                    <>
                      <div className="flex items-start">
                        {models.length > 3 && (
                          <button
                            type="button"
                            aria-label={expandedAccordions[info.row.id] ? "Collapse models" : "Expand models"}
                            className="rounded-sm text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() =>
                              setExpandedAccordions((prev) => ({
                                ...prev,
                                [info.row.id]: !prev[info.row.id],
                              }))
                            }
                          >
                            {expandedAccordions[info.row.id] ? (
                              <ChevronDown className="size-4" />
                            ) : (
                              <ChevronRight className="size-4" />
                            )}
                          </button>
                        )}
                        <div className="flex flex-wrap gap-1">
                          {models.slice(0, 3).map((model, index) =>
                            model === "all-proxy-models" ? (
                              <Badge key={index} variant="destructive">
                                All Proxy Models
                              </Badge>
                            ) : (
                              <Badge key={index}>
                                {model.length > 30
                                  ? `${getModelDisplayName(model).slice(0, 30)}...`
                                  : getModelDisplayName(model)}
                              </Badge>
                            ),
                          )}
                          {models.length > 3 && !expandedAccordions[info.row.id] && (
                            <Badge variant="secondary">
                              +{models.length - 3} {models.length - 3 === 1 ? "more model" : "more models"}
                            </Badge>
                          )}
                          {expandedAccordions[info.row.id] && (
                            <div className="flex flex-wrap gap-1">
                              {models.slice(3).map((model, index) =>
                                model === "all-proxy-models" ? (
                                  <Badge key={index + 3} variant="destructive">
                                    All Proxy Models
                                  </Badge>
                                ) : (
                                  <Badge key={index + 3}>
                                    {model.length > 30
                                      ? `${getModelDisplayName(model).slice(0, 30)}...`
                                      : getModelDisplayName(model)}
                                  </Badge>
                                ),
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              ) : null}
            </div>
          );
        },
      },
      {
        id: "rate_limits",
        header: "Rate Limits",
        size: 140,
        enableSorting: false,
        cell: ({ row }) => {
          const key = row.original;
          return (
            <div>
              <div>TPM: {key.tpm_limit !== null ? key.tpm_limit : "Unlimited"}</div>
              <div>RPM: {key.rpm_limit !== null ? key.rpm_limit : "Unlimited"}</div>
            </div>
          );
        },
      },
    ],
    [expandedAccordions, setSelectedKeyId],
  );

  if (selectedKeyId) {
    return (
      <div className="w-full">
        {selectedKey || selectedKeyLoadFailed ? (
          <KeyInfoView
            keyId={selectedKeyId}
            onClose={() => void setSelectedKeyId(null)}
            keyData={selectedKey}
            teams={[currentTeam]}
            onDelete={refetch}
            onKeyDataUpdate={handleSelectedKeyDataUpdate}
          />
        ) : (
          <div className="p-4 text-sm text-muted-foreground">Loading key...</div>
        )}
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="py-4">
        <DataTable
          data={displayKeys}
          columns={columns}
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
          isLoading={isLoading || isFetching}
          loadingMessage="Loading keys..."
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
                filterLabels={{ user_id: "User ID", key_hash: "Key ID" }}
              />
              <DataTableFilterDrawer
                table={table}
                open={filtersOpen}
                onOpenChange={setFiltersOpen}
                title="Filters"
                description={`Narrow down keys for ${teamAlias ?? "this team"}`}
              >
                {({ get, set }) => (
                  <>
                    <DataTableFilterField label="User ID">
                      <Input
                        value={(get("user_id") as string) ?? ""}
                        onChange={(event) => set("user_id", event.target.value)}
                        placeholder="Filter by user ID…"
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
    </div>
  );
}
