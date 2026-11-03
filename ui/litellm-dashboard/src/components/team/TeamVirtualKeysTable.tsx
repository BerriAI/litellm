"use client";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useTeamMemberBudgets } from "@/app/(dashboard)/hooks/teams/useTeamMemberBudgets";
import { teamMemberBudgetGate } from "@/components/shared/InheritedBudgetHint";
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
  DataTableMultiSortHeader,
  DataTableToolbar,
} from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { orgDetailHref, userDetailHref } from "@/utils/entityLinks";
import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnDef, ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { ChevronDown, ChevronRight } from "lucide-react";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { deriveKeyModelScope } from "../key_scope";
import { KEY_RATE_LIMIT_SORT_FIELDS } from "../key_team_helpers/key_sorting";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import { Organization } from "../networking";
import KeyInfoView from "../templates/key_info_view";

interface TeamVirtualKeysTableProps {
  teamId: string;
  teamAlias?: string;
  organization: Organization | null;
}

/**
 * TeamVirtualKeysTable – variant of VirtualKeysTable scoped to a single team.
 * Displays all virtual keys belonging to the team with same format and styling.
 */
const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

export function TeamVirtualKeysTable({ teamId, teamAlias, organization }: TeamVirtualKeysTableProps) {
  const { t } = useTranslation();
  const [selectedKey, setSelectedKey] = useState<KeyResponse | null>(null);
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [tablePagination, setTablePagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 50,
  });
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });

  const handleSearchChange = useCallback((value: string) => {
    setSearchInput(value);
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  const getFilterValue = useCallback(
    (columnId: string): string | undefined => {
      const entry = columnFilters.find((filter) => filter.id === columnId);
      return typeof entry?.value === "string" && entry.value.trim() ? entry.value.trim() : undefined;
    },
    [columnFilters],
  );

  const sortBy = sorting.length > 0 ? sorting[0].id : "created_at";
  const sortOrder = sorting.length > 0 ? (sorting[0].desc ? "desc" : "asc") : "desc";

  const pageIndex = tablePagination.pageIndex;
  const pageSize = tablePagination.pageSize;

  const keyListOptions = {
    teamID: teamId,
    search: searchQuery.trim() || undefined,
    userID: getFilterValue("user_id"),
    keyHash: getFilterValue("key_hash"),
    sortBy: sortBy || undefined,
    sortOrder: sortOrder || undefined,
    expand: "user",
  };

  const { data: keys, isPending: isLoading, isFetching, refetch } = useKeys(pageIndex + 1, pageSize, keyListOptions);

  const displayKeys = useMemo(() => keys?.keys ?? [], [keys?.keys]);

  const teamMemberBudgetTeamIds = useMemo(
    () => (displayKeys.some((key) => key.max_budget == null && Boolean(key.user_id)) ? [teamId] : []),
    [displayKeys, teamId],
  );
  const teamMemberBudgets = useTeamMemberBudgets(teamMemberBudgetTeamIds);

  const rowCount = keys?.total_count ?? 0;
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

  const handleColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>((updaterOrValue) => {
    setColumnFilters(updaterOrValue);
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  const columns: ColumnDef<KeyResponse>[] = useMemo(
    () => [
      {
        id: "token",
        accessorKey: "token",
        meta: { title: t("teams.virtualKeysTable.columns.keyId") },
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={t("teams.virtualKeysTable.columns.keyId")}
            variant="header-cycle"
          />
        ),
        size: 120,
        enableSorting: true,
        cell: (info) => (
          <IdCell value={info.getValue() as string | null} onClick={() => setSelectedKey(info.row.original)} />
        ),
      },
      {
        id: "key_alias",
        accessorKey: "key_alias",
        meta: { title: t("teams.virtualKeysTable.columns.keyAlias") },
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={t("teams.virtualKeysTable.columns.keyAlias")}
            variant="header-cycle"
          />
        ),
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
        header: t("teams.virtualKeysTable.columns.secretKey"),
        size: 120,
        enableSorting: false,
        cell: (info) => <span className="font-mono text-xs">{info.getValue() as string}</span>,
      },
      {
        id: "organization_id",
        accessorFn: (key) => (key.organization_id ?? key.org_id) || organization?.organization_id || null,
        header: t("teams.virtualKeysTable.columns.organizationId"),
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
        header: t("teams.virtualKeysTable.columns.userEmail"),
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
        header: t("teams.virtualKeysTable.columns.userId"),
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
        meta: { title: t("teams.virtualKeysTable.columns.createdAt") },
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={t("teams.virtualKeysTable.columns.createdAt")}
            variant="header-cycle"
          />
        ),
        size: 120,
        enableSorting: true,
        cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" />,
      },
      {
        id: "created_by",
        accessorKey: "created_by",
        header: t("teams.virtualKeysTable.columns.createdBy"),
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
        meta: { title: t("teams.virtualKeysTable.columns.updatedAt") },
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={t("teams.virtualKeysTable.columns.updatedAt")}
            variant="header-cycle"
          />
        ),
        size: 120,
        enableSorting: true,
        cell: (info) => (
          <DateCell
            value={info.getValue() as string | null}
            precision="date"
            fallback={t("teams.virtualKeysTable.dates.never")}
          />
        ),
      },
      {
        id: "last_active",
        accessorKey: "last_active",
        meta: { title: t("teams.virtualKeysTable.columns.lastActive") },
        header: ({ column }) => (
          <DataTableSortHeader column={column} title={t("teams.virtualKeysTable.columns.lastActive")} />
        ),
        size: 130,
        enableSorting: true,
        sortDescFirst: false,
        cell: (info) => (
          <DateCell
            value={info.getValue() as string | null}
            precision="date"
            fallback={t("teams.virtualKeysTable.dates.unknown")}
          />
        ),
      },
      {
        id: "expires",
        accessorKey: "expires",
        meta: { title: t("teams.virtualKeysTable.columns.expires") },
        header: ({ column }) => (
          <DataTableSortHeader column={column} title={t("teams.virtualKeysTable.columns.expires")} />
        ),
        size: 120,
        enableSorting: true,
        sortDescFirst: false,
        cell: (info) => (
          <DateCell
            value={info.getValue() as string | null}
            precision="date"
            fallback={t("teams.virtualKeysTable.dates.never")}
          />
        ),
      },
      {
        id: "spend",
        accessorKey: "spend",
        meta: { title: t("teams.virtualKeysTable.columns.spend"), numeric: true },
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={t("teams.virtualKeysTable.columns.spend")}
            variant="header-cycle"
          />
        ),
        size: 100,
        enableSorting: true,
        cell: (info) => <MoneyCell value={info.getValue() as number | null} decimals={4} />,
      },
      {
        id: "total_spend",
        accessorKey: "total_spend",
        meta: { title: t("teams.virtualKeysTable.columns.lifetimeSpend"), numeric: true },
        header: ({ column }) => (
          <DataTableSortHeader column={column} title={t("teams.virtualKeysTable.columns.lifetimeSpend")} />
        ),
        size: 130,
        enableSorting: true,
        sortDescFirst: true,
        cell: (info) => <MoneyCell value={info.getValue() as number | null | undefined} showZero />,
      },
      {
        id: "max_budget",
        accessorKey: "max_budget",
        meta: { title: t("teams.virtualKeysTable.columns.budget"), numeric: true },
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={t("teams.virtualKeysTable.columns.budget")}
            variant="header-cycle"
          />
        ),
        size: 160,
        enableSorting: true,
        cell: (info) => {
          const key = info.row.original;
          const memberGate =
            key.max_budget == null
              ? teamMemberBudgetGate(teamMemberBudgets[teamId], key.user_id, key.user?.user_email)
              : null;

          return memberGate ? (
            <span className="block w-full whitespace-nowrap text-right tabular-nums">
              ${formatNumberWithCommas(memberGate.maxBudget, 0)}{" "}
              <span className="text-muted-foreground">{t("teams.virtualKeysTable.budget.teamMember")}</span>
            </span>
          ) : (
            <MoneyCell
              value={info.getValue() as number | null}
              decimals={0}
              emptyText={t("teams.virtualKeysTable.budget.unlimited")}
              showZero
            />
          );
        },
      },
      {
        id: "budget_reset_at",
        accessorKey: "budget_reset_at",
        meta: { title: t("teams.virtualKeysTable.columns.budgetReset") },
        header: ({ column }) => (
          <DataTableSortHeader column={column} title={t("teams.virtualKeysTable.columns.budgetReset")} />
        ),
        size: 130,
        enableSorting: true,
        sortDescFirst: false,
        cell: (info) => (
          <DateCell value={info.getValue() as string | null} fallback={t("teams.virtualKeysTable.dates.never")} />
        ),
      },
      {
        id: "models",
        accessorKey: "models",
        header: t("teams.virtualKeysTable.columns.models"),
        size: 200,
        enableSorting: false,
        cell: (info) => {
          const models = info.getValue() as string[];
          const scope = deriveKeyModelScope(info.row.original.allowed_routes, info.row.original.key_type);
          const emptyModelsBadge = !scope.hasModelAccess ? (
            <SimpleTooltip content={t("teams.virtualKeysTable.models.noModelAccessTooltip", { scope: scope.label })}>
              <Badge variant="secondary" className="mb-1">
                {t("teams.virtualKeysTable.models.noModelAccess")}
              </Badge>
            </SimpleTooltip>
          ) : (
            <Badge variant="destructive" className="mb-1">
              {t("teams.virtualKeysTable.models.allProxyModels")}
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
                            aria-label={
                              expandedAccordions[info.row.id]
                                ? t("teams.virtualKeysTable.models.collapse")
                                : t("teams.virtualKeysTable.models.expand")
                            }
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
                                {t("teams.virtualKeysTable.models.allProxyModels")}
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
                              {t("teams.virtualKeysTable.models.more", { count: models.length - 3 })}
                            </Badge>
                          )}
                          {expandedAccordions[info.row.id] && (
                            <div className="flex flex-wrap gap-1">
                              {models.slice(3).map((model, index) =>
                                model === "all-proxy-models" ? (
                                  <Badge key={index + 3} variant="destructive">
                                    {t("teams.virtualKeysTable.models.allProxyModels")}
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
        meta: { title: t("teams.virtualKeysTable.columns.rateLimits") },
        header: ({ table }) => <DataTableMultiSortHeader table={table} fields={KEY_RATE_LIMIT_SORT_FIELDS} />,
        size: 140,
        enableSorting: false,
        cell: ({ row }) => {
          const key = row.original;
          return (
            <div>
              <div>
                {t("teams.virtualKeysTable.rateLimits.tpm")}
                {key.tpm_limit !== null ? key.tpm_limit : t("teams.virtualKeysTable.rateLimits.unlimited")}
              </div>
              <div>
                {t("teams.virtualKeysTable.rateLimits.rpm")}
                {key.rpm_limit !== null ? key.rpm_limit : t("teams.virtualKeysTable.rateLimits.unlimited")}
              </div>
            </div>
          );
        },
      },
    ],
    [expandedAccordions, teamMemberBudgets, teamId, organization?.organization_id, t],
  );

  const handleSortingChange = useCallback((updaterOrValue: React.SetStateAction<SortingState>) => {
    setSorting(updaterOrValue);
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  return (
    <div className="w-full">
      {selectedKey ? (
        <KeyInfoView
          keyId={selectedKey.token}
          onClose={() => setSelectedKey(null)}
          keyData={selectedKey}
          teams={[currentTeam]}
          onDelete={refetch}
        />
      ) : (
        <div className="py-4">
          <DataTable
            data={displayKeys}
            columns={columns}
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
            isLoading={isLoading || isFetching}
            loadingMessage={t("teams.virtualKeysTable.loading")}
            size="compact"
            toolbar={(table) => (
              <>
                <DataTableToolbar
                  table={table}
                  searchValue={searchInput}
                  onSearchChange={handleSearchChange}
                  searchPlaceholder={t("teams.virtualKeysTable.searchPlaceholder")}
                  onRefresh={() => refetch?.()}
                  isRefreshing={isFetching}
                  onOpenFilters={() => setFiltersOpen(true)}
                  filterLabels={{
                    user_id: t("teams.virtualKeysTable.filters.userId"),
                    key_hash: t("teams.virtualKeysTable.filters.keyId"),
                  }}
                />
                <DataTableFilterDrawer
                  table={table}
                  open={filtersOpen}
                  onOpenChange={setFiltersOpen}
                  title={t("teams.virtualKeysTable.filters.title")}
                  description={t("teams.virtualKeysTable.filters.description", {
                    team: teamAlias ?? t("teams.virtualKeysTable.filters.thisTeam"),
                  })}
                >
                  {({ get, set }) => (
                    <>
                      <DataTableFilterField label={t("teams.virtualKeysTable.filters.userId")}>
                        <Input
                          value={(get("user_id") as string) ?? ""}
                          onChange={(event) => set("user_id", event.target.value)}
                          placeholder={t("teams.virtualKeysTable.filters.userIdPlaceholder")}
                        />
                      </DataTableFilterField>
                      <DataTableFilterField label={t("teams.virtualKeysTable.filters.keyId")}>
                        <Input
                          value={(get("key_hash") as string) ?? ""}
                          onChange={(event) => set("key_hash", event.target.value)}
                          placeholder={t("teams.virtualKeysTable.filters.keyIdPlaceholder")}
                        />
                      </DataTableFilterField>
                    </>
                  )}
                </DataTableFilterDrawer>
              </>
            )}
          />
        </div>
      )}
    </div>
  );
}
