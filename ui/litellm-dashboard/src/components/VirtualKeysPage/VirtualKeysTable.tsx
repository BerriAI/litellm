"use client";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ChevronDownIcon, ChevronRightIcon, ChevronUpIcon, SwitchVerticalIcon } from "@heroicons/react/outline";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  PaginationState,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  Badge,
  Button,
  Icon,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
} from "@tremor/react";
import { InfoCircleOutlined, SyncOutlined } from "@ant-design/icons";
import { Button as AntButton, Popover, Skeleton, Tooltip, Typography } from "antd";
import React, { useEffect, useDeferredValue, useMemo, useState } from "react";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { useFilterLogic } from "../key_team_helpers/filter_logic";
import { PaginatedKeyAliasSelect } from "../KeyAliasSelect/PaginatedKeyAliasSelect/PaginatedKeyAliasSelect";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import FilterComponent, { FilterOption } from "../molecules/filter";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import { Organization } from "../networking";
import KeyInfoView from "../templates/key_info_view";

interface VirtualKeysTableProps {
  teams: Team[] | null;
  organizations: Organization[] | null;
  onSortChange?: (sortBy: string, sortOrder: "asc" | "desc") => void;
  currentSort?: {
    sortBy: string;
    sortOrder: "asc" | "desc";
  };
}

/**
 * VirtualKeysTable – a new table for keys that mimics the table styling used in view_logs.
 * The team selector and filtering have been removed so that all keys are shown.
 */

export function VirtualKeysTable({ teams, organizations, onSortChange, currentSort }: VirtualKeysTableProps) {
  const { data: fetchedOrganizations } = useOrganizations();
  const resolvedOrganizations = fetchedOrganizations ?? organizations ?? [];
  const [selectedKey, setSelectedKey] = useState<KeyResponse | null>(null);
  const [sorting, setSorting] = React.useState<SortingState>(() => {
    if (currentSort) {
      return [
        {
          id: currentSort.sortBy,
          desc: currentSort.sortOrder === "desc",
        },
      ];
    }
    return [
      {
        id: "created_at",
        desc: true,
      },
    ];
  });
  const [tablePagination, setTablePagination] = React.useState<PaginationState>({
    pageIndex: 0,
    pageSize: 50,
  });

  // Extract sort parameters from sorting state
  const sortBy = sorting.length > 0 ? sorting[0].id : null;
  const sortOrder = sorting.length > 0 ? (sorting[0].desc ? "desc" : "asc") : null;

  const {
    data: keys,
    isPending: isLoading,
    isFetching,
    isError,
    refetch,
  } = useKeys(tablePagination.pageIndex + 1, tablePagination.pageSize, {
    sortBy: sortBy || undefined,
    sortOrder: sortOrder || undefined,
    expand: "user",
  });
  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});

  // Use the filter logic hook

  const { filters, filteredKeys, filteredTotalCount, allTeams, allOrganizations, handleFilterChange, handleFilterReset } =
    useFilterLogic({
      keys: keys?.keys || [],
      teams,
      organizations,
    });

  // Defer the transition so the button stays in loading state until the table
  // has rendered with the new data (mirrors the spend-logs pattern)
  const isFetchingDeferred = useDeferredValue(isFetching);
  const isButtonLoading = (isFetching || isFetchingDeferred) && !isError;

  const handleRefresh = () => {
    refetch();
  };

  const totalCount = filteredTotalCount ?? keys?.total_count ?? 0;

  // Add a useEffect to call refresh when a key is created
  useEffect(() => {
    if (refetch) {
      const handleStorageChange = () => {
        refetch();
      };

      // Listen for storage events that might indicate a key was created
      window.addEventListener("storage", handleStorageChange);

      return () => {
        window.removeEventListener("storage", handleStorageChange);
      };
    }
  }, [refetch]);

  const columns: ColumnDef<KeyResponse>[] = useMemo(() => [
    {
      id: "expander",
      header: () => null,
      size: 40,
      enableSorting: false,
      cell: ({ row }) =>
        row.getCanExpand() ? (
          <button onClick={row.getToggleExpandedHandler()} style={{ cursor: "pointer" }}>
            {row.getIsExpanded() ? "▼" : "▶"}
          </button>
        ) : null,
    },
    {
      id: "token",
      accessorKey: "token",
      header: "Key ID",
      size: 100,
      enableSorting: true,
      cell: (info) => {
        const value = info.getValue() as string;
        const width = info.cell.column.getSize();
        return (
          <Tooltip title={value}>
            <Button
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate block"
              style={{ maxWidth: width, overflow: "hidden" }}
              onClick={() => setSelectedKey(info.row.original)}
            >
              {value ?? "-"}
            </Button>
          </Tooltip>
        );
      },
    },
    {
      id: "key_alias",
      accessorKey: "key_alias",
      header: "Key Alias",
      size: 150,
      enableSorting: true,
      cell: (info) => {
        const value = info.getValue() as string;
        const width = info.cell.column.getSize();
        return (
          <span className="font-mono text-xs truncate block" style={{ maxWidth: width, overflow: "hidden" }}>
            {value ?? "-"}
          </span>
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
      id: "team_alias",
      accessorKey: "team_id",
      header: "Team",
      size: 120,
      enableSorting: false,
      cell: (info) => {
        const teamId = info.getValue() as string | null;
        if (!teamId) return "-";
        const team = teams?.find((t) => t.team_id === teamId);
        const displayValue = team?.team_alias || teamId;
        const width = info.cell.column.getSize();
        return (
          <span className="font-mono text-xs truncate block" style={{ maxWidth: width, overflow: "hidden" }}>
            {displayValue}
          </span>
        );
      },
    },
    {
      id: "organization_alias",
      accessorKey: "org_id",
      header: "Organization",
      size: 140,
      enableSorting: false,
      cell: (info) => {
        const orgId = info.getValue() as string | null;
        if (!orgId) return "-";
        const org = resolvedOrganizations.find((o) => o.organization_id === orgId);
        const displayValue = org?.organization_alias || orgId;
        const width = info.cell.column.getSize();
        return (
          <span className="font-mono text-xs truncate block" style={{ maxWidth: width, overflow: "hidden" }}>
            {displayValue}
          </span>
        );
      },
    },
    {
      id: "user",
      accessorKey: "user",
      header: () => (
        <span className="flex items-center gap-1">
          User
          <Popover
            content="Displays the first available value: User Alias, User Email, or User ID."
            trigger="hover"
          >
            <InfoCircleOutlined className="text-gray-400 text-xs cursor-help" />
          </Popover>
        </span>
      ),
      size: 160,
      enableSorting: false,
      cell: ({ row }) => {
        const key = row.original;
        const userAlias = key.user?.user_alias ?? null;
        const userEmail = key.user?.user_email ?? key.user_email ?? null;
        const userId = key.user_id ?? null;
        const isDefaultAdmin = userId === "default_user_id";
        const displayValue = userAlias || userEmail || userId;
        const width = 160;

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
                  <Typography.Text
                    className="font-mono text-xs"
                    ellipsis={{ tooltip: value }}
                    copyable
                  >
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
              {displayValue || "-"}
            </span>
          </Popover>
        );
      },
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      header: "Created At",
      size: 120,
      enableSorting: true,
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "-";
      },
    },
    {
      id: "created_by",
      accessorKey: "created_by",
      header: "Created By",
      size: 160,
      enableSorting: false,
      cell: (info) => {
        const userId = info.getValue() as string | null;
        if (!userId) return "-";
        const isDefaultAdmin = userId === "default_user_id";
        const width = 160;

        const popoverContent = (
          <div className="flex flex-col gap-2 text-xs min-w-[200px] max-w-[300px]">
            <div className="flex flex-col min-w-0">
              <span className="text-gray-400">User ID</span>
              <Typography.Text
                className="font-mono text-xs"
                ellipsis={{ tooltip: userId }}
                copyable
              >
                {userId}
              </Typography.Text>
            </div>
          </div>
        );

        if (isDefaultAdmin) {
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
              {userId}
            </span>
          </Popover>
        );
      },
    },
    {
      id: "updated_at",
      accessorKey: "updated_at",
      header: "Updated At",
      size: 120,
      enableSorting: true,
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "Never";
      },
    },
    {
      id: "last_active",
      accessorKey: "last_active",
      header: () => (
        <span className="flex items-center gap-1">
          Last Active
          <Popover
            content="This is a new field and is not backfilled. Only new key usage will update this value."
            trigger="hover"
          >
            <InfoCircleOutlined className="text-gray-400 text-xs cursor-help" />
          </Popover>
        </span>
      ),
      size: 130,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue();
        if (!value) return "Unknown";
        const date = new Date(value as string);
        return (
          <Tooltip title={date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "long" })}>
            <span>{date.toLocaleDateString()}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "expires",
      accessorKey: "expires",
      header: "Expires",
      size: 120,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "Never";
      },
    },
    {
      id: "spend",
      accessorKey: "spend",
      header: "Spend (USD)",
      size: 100,
      enableSorting: true,
      cell: (info) => formatNumberWithCommas(info.getValue() as number, 4),
    },
    {
      id: "max_budget",
      accessorKey: "max_budget",
      header: "Budget (USD)",
      size: 110,
      enableSorting: true,
      cell: (info) => {
        const maxBudget = info.getValue() as number | null;
        if (maxBudget === null) {
          return "Unlimited";
        }
        return `$${formatNumberWithCommas(maxBudget)}`;
      },
    },
    {
      id: "budget_reset_at",
      accessorKey: "budget_reset_at",
      header: "Budget Reset",
      size: 130,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleString() : "Never";
      },
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
                  <Badge size={"xs"} className="mb-1" color="red">
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
                            onClick={() => {
                              setExpandedAccordions((prev) => ({
                                ...prev,
                                [info.row.id]: !prev[info.row.id],
                              }));
                            }}
                          />
                        </div>
                      )}
                      <div className="flex flex-wrap gap-1">
                        {models.slice(0, 3).map((model, index) =>
                          model === "all-proxy-models" ? (
                            <Badge key={index} size={"xs"} color="red">
                              <Text>All Proxy Models</Text>
                            </Badge>
                          ) : (
                            <Badge key={index} size={"xs"} color="blue">
                              <Text>
                                {model.length > 30
                                  ? `${getModelDisplayName(model).slice(0, 30)}...`
                                  : getModelDisplayName(model)}
                              </Text>
                            </Badge>
                          ),
                        )}
                        {models.length > 3 && !expandedAccordions[info.row.id] && (
                          <Badge size={"xs"} color="gray" className="cursor-pointer">
                            <Text>
                              +{models.length - 3} {models.length - 3 === 1 ? "more model" : "more models"}
                            </Text>
                          </Badge>
                        )}
                        {expandedAccordions[info.row.id] && (
                          <div className="flex flex-wrap gap-1">
                            {models.slice(3).map((model, index) =>
                              model === "all-proxy-models" ? (
                                <Badge key={index + 3} size={"xs"} color="red">
                                  <Text>All Proxy Models</Text>
                                </Badge>
                              ) : (
                                <Badge key={index + 3} size={"xs"} color="blue">
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
  ], [teams, resolvedOrganizations]);

  const filterOptions: FilterOption[] = [
    {
      name: "Team ID",
      label: "Team ID",
      isSearchable: true,
      searchFn: async (searchText: string) => {
        if (!allTeams || allTeams.length === 0) return [];

        const filteredTeams = allTeams.filter(
          (team) =>
            team.team_id.toLowerCase().includes(searchText.toLowerCase()) ||
            (team.team_alias && team.team_alias.toLowerCase().includes(searchText.toLowerCase())),
        );

        return filteredTeams.map((team) => ({
          label: `${team.team_alias || team.team_id} (${team.team_id})`,
          value: team.team_id,
        }));
      },
    },
    {
      name: "Organization ID",
      label: "Organization ID",
      isSearchable: true,
      searchFn: async (searchText: string) => {
        if (!allOrganizations || allOrganizations.length === 0) return [];

        const filteredOrgs = allOrganizations.filter(
          (org) => org.organization_id?.toLowerCase().includes(searchText.toLowerCase()) ?? false,
        );

        return filteredOrgs
          .filter((org) => org.organization_id !== null && org.organization_id !== undefined)
          .map((org) => ({
            label: `${org.organization_id || "Unknown"} (${org.organization_id})`,
            value: org.organization_id as string,
          }));
      },
    },
    {
      name: "Key Alias",
      label: "Key Alias",
      customComponent: PaginatedKeyAliasSelect,
    },
    {
      name: "User ID",
      label: "User ID",
      isSearchable: false,
    },
    {
      name: "Key Hash",
      label: "Key Hash",
      isSearchable: false,
    },
  ];

  const table = useReactTable({
    data: filteredKeys,
    columns: columns.filter((col) => col.id !== "expander"),
    columnResizeMode: "onChange",
    columnResizeDirection: "ltr",
    state: {
      sorting,
      pagination: tablePagination,
    },
    onSortingChange: (updaterOrValue) => {
      const newSorting = typeof updaterOrValue === "function" ? updaterOrValue(sorting) : updaterOrValue;
      setSorting(newSorting);
      if (newSorting && newSorting.length > 0) {
        const sortState = newSorting[0];
        const sortBy = sortState.id;
        const sortOrder = sortState.desc ? "desc" : "asc";
        // Update filters state without triggering debouncedSearch
        // The useKeys hook will automatically refetch with the new sort parameters
        handleFilterChange(
          {
            ...filters,
            "Sort By": sortBy,
            "Sort Order": sortOrder,
          },
          true, // skipDebounce - let useKeys handle the API call with correct page size
        );
        onSortChange?.(sortBy, sortOrder);
      }
    },
    onPaginationChange: setTablePagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    enableSorting: true,
    manualSorting: false,
    manualPagination: true,
    pageCount: Math.ceil(totalCount / tablePagination.pageSize),
  });

  // Update local sorting state when currentSort prop changes
  React.useEffect(() => {
    if (currentSort) {
      setSorting([
        {
          id: currentSort.sortBy,
          desc: currentSort.sortOrder === "desc",
        },
      ]);
    }
  }, [currentSort]);

  const { pageIndex, pageSize } = table.getState().pagination;
  const start = pageIndex * pageSize + 1;
  const end = Math.min((pageIndex + 1) * pageSize, totalCount);
  const rangeLabel = `${start} - ${end}`;
  return (
    <div className="w-full h-full overflow-hidden">
      {selectedKey ? (
        <KeyInfoView
          keyId={selectedKey.token}
          onClose={() => setSelectedKey(null)}
          keyData={selectedKey}
          teams={allTeams}
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

          <div className="flex items-center justify-between w-full mb-4">
            <div className="inline-flex items-center gap-2">
              {isLoading ? (
                <Skeleton.Node active style={{ width: 200, height: 20 }} />
              ) : (
                <span className="inline-flex text-sm text-gray-700">
                  Showing {rangeLabel} of {totalCount} results
                </span>
              )}

              <AntButton
                type="default"
                icon={<SyncOutlined spin={isButtonLoading} />}
                onClick={handleRefresh}
                disabled={isButtonLoading}
                title="Fetch data"
              >
                {isButtonLoading ? "Fetching" : "Fetch"}
              </AntButton>
            </div>

            <div className="inline-flex items-center gap-2">
              {isLoading ? (
                <Skeleton.Node active style={{ width: 74, height: 20 }} />
              ) : (
                <span className="text-sm text-gray-700">
                  Page {pageIndex + 1} of {table.getPageCount()}
                </span>
              )}

              {isLoading ? (
                <Skeleton.Button active size="small" style={{ width: 84, height: 30 }} />
              ) : (
                <button
                  onClick={() => table.previousPage()}
                  disabled={isLoading || !table.getCanPreviousPage()}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
              )}

              {isLoading ? (
                <Skeleton.Button active size="small" style={{ width: 58, height: 30 }} />
              ) : (
                <button
                  onClick={() => table.nextPage()}
                  disabled={isLoading || !table.getCanNextPage()}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              )}
            </div>
          </div>
          <div className="h-[75vh] overflow-auto">
            <div className="rounded-lg custom-border relative">
              <div className="overflow-x-auto">
                <Table className="[&_td]:py-0.5 [&_th]:py-1" style={{ width: table.getCenterTotalSize() }}>
                  <TableHead>
                    {table.getHeaderGroups().map((headerGroup) => (
                      <TableRow key={headerGroup.id}>
                        {headerGroup.headers.map((header) => (
                          <TableHeaderCell
                            key={header.id}
                            data-header-id={header.id}
                            className={`py-1 h-8 relative hover:bg-gray-50 ${header.id === "actions"
                              ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                              : ""
                              }`}
                            style={{
                              width: header.getSize(),
                              position: "relative",
                              cursor: header.column.getCanSort() ? "pointer" : "default",
                            }}
                            onMouseEnter={() => {
                              const resizer = document.querySelector(`[data-header-id="${header.id}"] .resizer`);
                              if (resizer) {
                                (resizer as HTMLElement).style.opacity = "0.5";
                              }
                            }}
                            onMouseLeave={() => {
                              const resizer = document.querySelector(`[data-header-id="${header.id}"] .resizer`);
                              if (resizer && !header.column.getIsResizing()) {
                                (resizer as HTMLElement).style.opacity = "0";
                              }
                            }}
                            onClick={header.column.getCanSort() ? header.column.getToggleSortingHandler() : undefined}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center">
                                {header.isPlaceholder
                                  ? null
                                  : flexRender(header.column.columnDef.header, header.getContext())}
                              </div>
                              {header.id !== "actions" && header.column.getCanSort() && (
                                <div className="w-4">
                                  {header.column.getIsSorted() ? (
                                    {
                                      asc: <ChevronUpIcon className="h-4 w-4 text-blue-500" />,
                                      desc: <ChevronDownIcon className="h-4 w-4 text-blue-500" />,
                                    }[header.column.getIsSorted() as string]
                                  ) : (
                                    <SwitchVerticalIcon className="h-4 w-4 text-gray-400" />
                                  )}
                                </div>
                              )}
                              <div
                                onDoubleClick={() => header.column.resetSize()}
                                onMouseDown={header.getResizeHandler()}
                                onTouchStart={header.getResizeHandler()}
                                className={`resizer ${table.options.columnResizeDirection} ${header.column.getIsResizing() ? "isResizing" : ""}`}
                                style={{
                                  position: "absolute",
                                  right: 0,
                                  top: 0,
                                  height: "100%",
                                  width: "5px",
                                  background: header.column.getIsResizing() ? "#3b82f6" : "transparent",
                                  cursor: "col-resize",
                                  userSelect: "none",
                                  touchAction: "none",
                                  opacity: header.column.getIsResizing() ? 1 : 0,
                                }}
                              />
                            </div>
                          </TableHeaderCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableHead>
                  <TableBody>
                    {isLoading ? (
                      <TableRow>
                        <TableCell colSpan={columns.length} className="h-8 text-center">
                          <div className="text-center text-gray-500">
                            <p>🚅 Loading keys...</p>
                          </div>
                        </TableCell>
                      </TableRow>
                    ) : filteredKeys.length > 0 ? (
                      table.getRowModel().rows.map((row) => (
                        <TableRow key={row.id} className="h-8">
                          {row.getVisibleCells().map((cell) => (
                            <TableCell
                              key={cell.id}
                              style={{
                                width: cell.column.getSize(),
                                maxWidth: "8-x",
                                whiteSpace: "pre-wrap",
                                overflow: "hidden",
                              }}
                              className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${cell.column.id === "models" && Array.isArray(cell.getValue()) && (cell.getValue() as string[]).length > 3 ? "px-0" : ""}`}
                            >
                              {flexRender(cell.column.columnDef.cell, cell.getContext())}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={columns.length} className="h-8 text-center">
                          <div className="text-center text-gray-500">
                            <p>No keys found</p>
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
