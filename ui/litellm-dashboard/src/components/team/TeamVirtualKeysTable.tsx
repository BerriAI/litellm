"use client";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { DateCell, IdCell, MoneyCell } from "@/components/shared/table_cells";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableSortHeader,
  DataTableToolbar,
} from "@/components/shared/DataTable";
import { Input } from "@/components/ui/input";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { ColumnDef, ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Badge, Icon, Text } from "@tremor/react";
import { Popover, Tooltip, Typography } from "antd";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { deriveKeyModelScope } from "../key_scope";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import { Organization } from "../networking";
import KeyInfoView from "../templates/key_info_view";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation("gateway");
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

  const {
    data: keys,
    isPending: isLoading,
    isFetching,
    refetch,
  } = useKeys(pageIndex + 1, pageSize, {
    teamID: teamId,
    selectedKeyAlias: searchQuery.trim() || undefined,
    userID: getFilterValue("user_id"),
    sortBy: sortBy || undefined,
    sortOrder: sortOrder || undefined,
    expand: "user",
  });

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
        meta: { title: t("virtualKeys.columns.keyId") },
        header: ({ column }) => (
          <DataTableSortHeader column={column} title={t("virtualKeys.columns.keyId")} variant="header-cycle" />
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
        meta: { title: t("teams.virtualKeys.keyAlias") },
        header: ({ column }) => (
          <DataTableSortHeader column={column} title={t("teams.virtualKeys.keyAlias")} variant="header-cycle" />
        ),
        size: 150,
        enableSorting: true,
        cell: (info) => {
          const value = info.getValue() as string;
          const width = info.cell.column.getSize();
          return (
            <Tooltip title={value}>
              <span className="font-mono text-xs truncate block" style={{ maxWidth: width, overflow: "hidden" }}>
                {value ?? "-"}
              </span>
            </Tooltip>
          );
        },
      },
      {
        id: "key_name",
        accessorKey: "key_name",
        header: t("teams.virtualKeys.secretKey"),
        size: 120,
        enableSorting: false,
        cell: (info) => <span className="font-mono text-xs">{info.getValue() as string}</span>,
      },
      {
        id: "organization_id",
        accessorKey: "organization_id",
        header: t("teams.details.settings.organizationId"),
        size: 140,
        enableSorting: false,
        cell: (info) => (info.getValue() ? info.renderValue() : "-"),
      },
      {
        id: "user_email",
        accessorKey: "user",
        header: t("virtualKeys.columns.userEmail"),
        size: 160,
        enableSorting: false,
        cell: (info) => {
          const user = info.getValue() as { user_email?: string } | undefined;
          const value = user?.user_email;
          const width = info.cell.column.getSize();
          return (
            <Tooltip title={value}>
              <span className="font-mono text-xs truncate block" style={{ maxWidth: width, overflow: "hidden" }}>
                {value ?? "-"}
              </span>
            </Tooltip>
          );
        },
      },
      {
        id: "user_id",
        accessorKey: "user_id",
        header: t("virtualKeys.columns.userId"),
        size: 70,
        enableSorting: false,
        cell: (info) => {
          const userId = info.getValue() as string | null;
          const displayValue = userId === "default_user_id" ? t("virtualKeys.values.defaultProxyAdmin") : userId;
          const width = info.cell.column.getSize();
          return (
            <Tooltip title={displayValue}>
              <span className="font-mono text-xs truncate block" style={{ maxWidth: width, overflow: "hidden" }}>
                {displayValue ?? "-"}
              </span>
            </Tooltip>
          );
        },
      },
      {
        id: "created_at",
        accessorKey: "created_at",
        meta: { title: t("virtualKeys.columns.createdAt") },
        header: ({ column }) => (
          <DataTableSortHeader column={column} title={t("virtualKeys.columns.createdAt")} variant="header-cycle" />
        ),
        size: 120,
        enableSorting: true,
        cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" />,
      },
      {
        id: "created_by",
        accessorKey: "created_by",
        header: t("virtualKeys.columns.createdBy"),
        size: 130,
        enableSorting: false,
        cell: (info) => {
          const userId = info.getValue() as string | null;
          if (!userId) return "-";
          const { created_by_user } = info.row.original;
          const userAlias = created_by_user?.user_alias ?? null;
          const userEmail = created_by_user?.user_email ?? null;
          const isDefaultAdmin = userId === "default_user_id";
          const displayValue = userAlias || userEmail || userId;
          const width = info.cell.column.getSize();

          const popoverContent = (
            <div className="flex flex-col gap-2 text-xs min-w-[200px] max-w-[300px]">
              {[
                { label: t("virtualKeys.columns.userAlias"), value: userAlias },
                { label: t("virtualKeys.columns.userEmail"), value: userEmail },
                { label: t("virtualKeys.columns.userId"), value: userId },
              ].map(({ label, value }) => (
                <div key={label} className="flex flex-col min-w-0">
                  <span className="text-gray-400">{label}</span>
                  {value ? (
                    <Typography.Text className="font-mono text-xs" ellipsis={{ tooltip: value }} copyable>
                      {value}
                    </Typography.Text>
                  ) : (
                    <span className="font-mono">-</span>
                  )}
                </div>
              ))}
            </div>
          );

          if (isDefaultAdmin && !userAlias && !userEmail) {
            return (
              <Popover content={popoverContent} trigger="hover" placement="bottomLeft">
                <span className="cursor-default">
                  <DefaultProxyAdminTag userId={userId} />
                </span>
              </Popover>
            );
          }

          return (
            <Popover content={popoverContent} trigger="hover" placement="bottomLeft">
              <span
                className="font-mono text-xs truncate block cursor-default"
                style={{ maxWidth: width, overflow: "hidden" }}
              >
                {displayValue}
              </span>
            </Popover>
          );
        },
      },
      {
        id: "updated_at",
        accessorKey: "updated_at",
        meta: { title: t("virtualKeys.columns.updatedAt") },
        header: ({ column }) => (
          <DataTableSortHeader column={column} title={t("virtualKeys.columns.updatedAt")} variant="header-cycle" />
        ),
        size: 120,
        enableSorting: true,
        cell: (info) => (
          <DateCell
            value={info.getValue() as string | null}
            precision="date"
            fallback={t("virtualKeys.values.never")}
          />
        ),
      },
      {
        id: "last_active",
        accessorKey: "last_active",
        header: t("virtualKeys.columns.lastActive"),
        size: 130,
        enableSorting: false,
        cell: (info) => (
          <DateCell
            value={info.getValue() as string | null}
            precision="date"
            fallback={t("virtualKeys.values.unknown")}
          />
        ),
      },
      {
        id: "expires",
        accessorKey: "expires",
        header: t("virtualKeys.columns.expires"),
        size: 120,
        enableSorting: false,
        cell: (info) => (
          <DateCell
            value={info.getValue() as string | null}
            precision="date"
            fallback={t("virtualKeys.values.never")}
          />
        ),
      },
      {
        id: "spend",
        accessorKey: "spend",
        meta: { title: `${t("virtualKeys.columns.spend")} (USD)` },
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={`${t("virtualKeys.columns.spend")} (USD)`}
            variant="header-cycle"
          />
        ),
        size: 100,
        enableSorting: true,
        cell: (info) => <MoneyCell value={info.getValue() as number | null} decimals={4} />,
      },
      {
        id: "max_budget",
        accessorKey: "max_budget",
        meta: { title: `${t("virtualKeys.columns.budget")} (USD)` },
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={`${t("virtualKeys.columns.budget")} (USD)`}
            variant="header-cycle"
          />
        ),
        size: 110,
        enableSorting: true,
        cell: (info) => (
          <MoneyCell
            value={info.getValue() as number | null}
            decimals={0}
            emptyText={t("virtualKeys.values.unlimited")}
            showZero
          />
        ),
      },
      {
        id: "budget_reset_at",
        accessorKey: "budget_reset_at",
        header: t("virtualKeys.columns.budgetReset"),
        size: 130,
        enableSorting: false,
        cell: (info) => <DateCell value={info.getValue() as string | null} fallback={t("virtualKeys.values.never")} />,
      },
      {
        id: "models",
        accessorKey: "models",
        header: t("virtualKeys.columns.models"),
        size: 200,
        enableSorting: false,
        cell: (info) => {
          const models = info.getValue() as string[];
          const scope = deriveKeyModelScope(info.row.original.allowed_routes, info.row.original.key_type);
          const emptyModelsBadge = !scope.hasModelAccess ? (
            <Tooltip title={t("virtualKeys.values.scopedRoutes", { scope: scope.label })}>
              <Badge size="xs" className="mb-1" color="gray">
                <Text>{t("virtualKeys.values.noModelAccess")}</Text>
              </Badge>
            </Tooltip>
          ) : (
            <Badge size="xs" className="mb-1" color="red">
              <Text>{t("virtualKeys.values.allProxyModels")}</Text>
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
                          <div>
                            <Icon
                              icon={expandedAccordions[info.row.id] ? ChevronDownIcon : ChevronRightIcon}
                              className="cursor-pointer"
                              size="xs"
                              onClick={() =>
                                setExpandedAccordions((prev) => ({
                                  ...prev,
                                  [info.row.id]: !prev[info.row.id],
                                }))
                              }
                            />
                          </div>
                        )}
                        <div className="flex flex-wrap gap-1">
                          {models.slice(0, 3).map((model, index) =>
                            model === "all-proxy-models" ? (
                              <Badge key={index} size="xs" color="red">
                                <Text>{t("virtualKeys.values.allProxyModels")}</Text>
                              </Badge>
                            ) : (
                              <Badge key={index} size="xs" color="blue">
                                <Text>
                                  {model.length > 30
                                    ? `${getModelDisplayName(model).slice(0, 30)}...`
                                    : getModelDisplayName(model)}
                                </Text>
                              </Badge>
                            ),
                          )}
                          {models.length > 3 && !expandedAccordions[info.row.id] && (
                            <Badge size="xs" color="gray" className="cursor-pointer">
                              <Text>{t("virtualKeys.values.more", { count: models.length - 3 })}</Text>
                            </Badge>
                          )}
                          {expandedAccordions[info.row.id] && (
                            <div className="flex flex-wrap gap-1">
                              {models.slice(3).map((model, index) =>
                                model === "all-proxy-models" ? (
                                  <Badge key={index + 3} size="xs" color="red">
                                    <Text>{t("virtualKeys.values.allProxyModels")}</Text>
                                  </Badge>
                                ) : (
                                  <Badge key={index + 3} size="xs" color="blue">
                                    <Text>
                                      {model.length > 30
                                        ? `${getModelDisplayName(model).slice(0, 30)}...`
                                        : getModelDisplayName(model)}
                                    </Text>
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
        header: t("virtualKeys.columns.rateLimits"),
        size: 140,
        enableSorting: false,
        cell: ({ row }) => {
          const key = row.original;
          return (
            <div>
              <div>TPM: {key.tpm_limit !== null ? key.tpm_limit : t("virtualKeys.values.unlimited")}</div>
              <div>RPM: {key.rpm_limit !== null ? key.rpm_limit : t("virtualKeys.values.unlimited")}</div>
            </div>
          );
        },
      },
    ],
    [expandedAccordions, t],
  );

  const handleSortingChange = useCallback((updaterOrValue: React.SetStateAction<SortingState>) => {
    setSorting(updaterOrValue);
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  return (
    <div className="w-full h-full overflow-hidden">
      {selectedKey ? (
        <KeyInfoView
          keyId={selectedKey.token}
          onClose={() => setSelectedKey(null)}
          keyData={selectedKey}
          teams={[currentTeam]}
          onDelete={refetch}
        />
      ) : (
        <div className="py-4 flex-1 overflow-hidden">
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
            loadingMessage={t("virtualKeys.loading")}
            maxBodyHeight="75vh"
            size="compact"
            toolbar={(table) => (
              <>
                <DataTableToolbar
                  table={table}
                  searchValue={searchInput}
                  onSearchChange={handleSearchChange}
                  searchPlaceholder={t("virtualKeys.searchPlaceholder")}
                  onRefresh={() => refetch?.()}
                  isRefreshing={isFetching}
                  onOpenFilters={() => setFiltersOpen(true)}
                  filterLabels={{ user_id: t("virtualKeys.columns.userId") }}
                />
                <DataTableFilterDrawer
                  table={table}
                  open={filtersOpen}
                  onOpenChange={setFiltersOpen}
                  title={t("virtualKeys.filters.title")}
                  description={t("teams.virtualKeys.filtersDescription", {
                    team: teamAlias ?? t("teams.virtualKeys.thisTeam"),
                  })}
                >
                  {({ get, set }) => (
                    <DataTableFilterField label={t("virtualKeys.columns.userId")}>
                      <Input
                        value={(get("user_id") as string) ?? ""}
                        onChange={(event) => set("user_id", event.target.value)}
                        placeholder={t("teams.virtualKeys.filterUserId")}
                      />
                    </DataTableFilterField>
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
