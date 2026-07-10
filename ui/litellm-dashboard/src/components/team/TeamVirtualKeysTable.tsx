"use client";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { DateCell, IdCell, MoneyCell } from "@/components/shared/table_cells";
import { DataTable, DataTablePagination, DataTableSortHeader } from "@/components/shared/DataTable";
import { ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { ColumnDef, PaginationState, SortingState } from "@tanstack/react-table";
import { Badge, Icon, Text } from "@tremor/react";
import { Popover, Tooltip, Typography } from "antd";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import FilterComponent, { FilterOption } from "../molecules/filter";
import { Organization } from "../networking";
import KeyInfoView from "../templates/key_info_view";
import { useQuery } from "@tanstack/react-query";
import { fetchTeamFilterOptions } from "../key_team_helpers/filter_helpers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

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
  const { accessToken } = useAuthorized();
  const [selectedKey, setSelectedKey] = useState<KeyResponse | null>(null);
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [tablePagination, setTablePagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 50,
  });
  const [filters, setFilters] = useState<Record<string, string>>({
    "Organization ID": "",
    "Key Alias": "",
    "User ID": "",
  });

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
    organizationID: filters["Organization ID"]?.trim() || undefined,
    selectedKeyAlias: filters["Key Alias"]?.trim() || undefined,
    userID: filters["User ID"]?.trim() || undefined,
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

  const teamFilterOptionsQuery = useQuery({
    queryKey: ["teamFilterOptions", teamId, accessToken],
    queryFn: async () => fetchTeamFilterOptions(accessToken, teamId),
    enabled: !!accessToken && !!teamId,
    staleTime: 30000, // 30 seconds - align with useKeys
  });
  const teamFilterOptions = teamFilterOptionsQuery.data || {
    keyAliases: [],
    organizationIds: [],
    userIds: [],
  };

  const handleStorageChange = useCallback(() => {
    refetch?.();
  }, [refetch]);

  useEffect(() => {
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, [handleStorageChange]);

  const handleFilterChange = useCallback((newFilters: Record<string, string>) => {
    setFilters((prev) => ({
      ...prev,
      "Organization ID": newFilters["Organization ID"] ?? prev["Organization ID"],
      "Key Alias": newFilters["Key Alias"] ?? prev["Key Alias"],
      "User ID": newFilters["User ID"] ?? prev["User ID"],
    }));
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  const handleFilterReset = useCallback(() => {
    setFilters({
      "Organization ID": "",
      "Key Alias": "",
      "User ID": "",
    });
    setSorting(DEFAULT_SORTING);
    setTablePagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  const filterOptions: FilterOption[] = useMemo(
    () => [
      {
        name: "Organization ID",
        label: "Organization ID",
        isSearchable: true,
        searchFn: async (searchText: string) => {
          const { organizationIds } = teamFilterOptions;
          if (!organizationIds.length) return [];
          const lower = searchText.toLowerCase();
          const filtered = lower ? organizationIds.filter((id) => id.toLowerCase().includes(lower)) : organizationIds;
          return filtered.map((id) => ({ label: id, value: id }));
        },
      },
      {
        name: "Key Alias",
        label: "Key Alias",
        isSearchable: true,
        searchFn: async (searchText: string) => {
          const { keyAliases } = teamFilterOptions;
          const lower = searchText.toLowerCase();
          const filtered = lower ? keyAliases.filter((alias) => alias.toLowerCase().includes(lower)) : keyAliases;
          return filtered.map((alias) => ({ label: alias, value: alias }));
        },
      },
      {
        name: "User ID",
        label: "User ID",
        isSearchable: true,
        searchFn: async (searchText: string) => {
          const { userIds } = teamFilterOptions;
          const lower = searchText.toLowerCase();
          const filtered = lower
            ? userIds.filter((u) => u.id.toLowerCase().includes(lower) || u.email.toLowerCase().includes(lower))
            : userIds;
          return filtered.map((u) => ({
            label: u.email ? `${u.id} (${u.email})` : u.id,
            value: u.id,
          }));
        },
      },
    ],
    [teamFilterOptions],
  );

  const columns: ColumnDef<KeyResponse>[] = useMemo(
    () => [
      {
        id: "token",
        accessorKey: "token",
        header: ({ column }) => <DataTableSortHeader column={column} title="Key ID" variant="header-cycle" />,
        size: 120,
        enableSorting: true,
        cell: (info) => (
          <IdCell value={info.getValue() as string | null} onClick={() => setSelectedKey(info.row.original)} />
        ),
      },
      {
        id: "key_alias",
        accessorKey: "key_alias",
        header: ({ column }) => <DataTableSortHeader column={column} title="Key Alias" variant="header-cycle" />,
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
        cell: (info) => (info.getValue() ? info.renderValue() : "-"),
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
        header: "User ID",
        size: 70,
        enableSorting: false,
        cell: (info) => {
          const userId = info.getValue() as string | null;
          const displayValue = userId === "default_user_id" ? "Default Proxy Admin" : userId;
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
          const userAlias = created_by_user?.user_alias ?? null;
          const userEmail = created_by_user?.user_email ?? null;
          const isDefaultAdmin = userId === "default_user_id";
          const displayValue = userAlias || userEmail || userId;
          const width = info.cell.column.getSize();

          const popoverContent = (
            <div className="flex flex-col gap-2 text-xs min-w-[200px] max-w-[300px]">
              {[
                { label: "User Alias", value: userAlias },
                { label: "User Email", value: userEmail },
                { label: "User ID", value: userId },
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
        header: ({ column }) => <DataTableSortHeader column={column} title="Spend (USD)" variant="header-cycle" />,
        size: 100,
        enableSorting: true,
        cell: (info) => <MoneyCell value={info.getValue() as number | null} decimals={4} />,
      },
      {
        id: "max_budget",
        accessorKey: "max_budget",
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
          return (
            <div className="flex flex-col py-2">
              {Array.isArray(models) ? (
                <div className="flex flex-col">
                  {models.length === 0 ? (
                    <Badge size="xs" className="mb-1" color="red">
                      <Text>All Proxy Models</Text>
                    </Badge>
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
                                <Text>All Proxy Models</Text>
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
                              <Text>
                                +{models.length - 3} {models.length - 3 === 1 ? "more model" : "more models"}
                              </Text>
                            </Badge>
                          )}
                          {expandedAccordions[info.row.id] && (
                            <div className="flex flex-wrap gap-1">
                              {models.slice(3).map((model, index) =>
                                model === "all-proxy-models" ? (
                                  <Badge key={index + 3} size="xs" color="red">
                                    <Text>All Proxy Models</Text>
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
    [expandedAccordions],
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
        <div className="border-b py-4 flex-1 overflow-hidden">
          <div className="w-full mb-6">
            <FilterComponent
              options={filterOptions}
              onApplyFilters={handleFilterChange}
              initialValues={filters}
              onResetFilters={handleFilterReset}
            />
          </div>

          <div className="w-full mb-4">
            <DataTablePagination
              page={pageIndex}
              pageSize={pageSize}
              rowCount={rowCount}
              onPageChange={(nextPage) => setTablePagination((prev) => ({ ...prev, pageIndex: nextPage }))}
              onPageSizeChange={(nextSize) => setTablePagination({ pageIndex: 0, pageSize: nextSize })}
              isLoading={isLoading || isFetching}
            />
          </div>

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
            paginationSlot={() => null}
            enableColumnResizing
            columnResizeMode="onChange"
            isLoading={isLoading || isFetching}
            loadingMessage="Loading keys..."
            noDataMessage="No keys found"
            maxBodyHeight="75vh"
            size="compact"
          />
        </div>
      )}
    </div>
  );
}
