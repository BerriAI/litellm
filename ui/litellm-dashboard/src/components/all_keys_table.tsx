"use client";
import React, { useEffect, useState } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { Select, SelectItem } from "@tremor/react";
import { Button } from "@tremor/react";
import KeyInfoView from "./templates/key_info_view";
import { Tooltip } from "antd";
import { Team, KeyResponse } from "./key_team_helpers/key_list";
import FilterComponent from "./molecules/filter";
import { FilterOption } from "./molecules/filter";
import { Organization, userListCall } from "./networking";
import { useFilterLogic } from "./key_team_helpers/filter_logic";
import { Setter } from "@/types";
import { updateExistingKeys } from "@/utils/dataUtils";
import { flexRender, getCoreRowModel, getSortedRowModel, SortingState, useReactTable } from "@tanstack/react-table";
import { Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell, Icon } from "@tremor/react";
import { SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { Badge, Text } from "@tremor/react";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { formatNumberWithCommas } from "@/utils/dataUtils";

interface AllKeysTableProps {
  keys: KeyResponse[];
  setKeys: (keys: KeyResponse[] | ((prev: KeyResponse[]) => KeyResponse[])) => void;
  isLoading?: boolean;
  pagination: {
    currentPage: number;
    totalPages: number;
    totalCount: number;
  };
  onPageChange: (page: number) => void;
  pageSize?: number;
  teams: Team[] | null;
  selectedTeam: Team | null;
  setSelectedTeam: (team: Team | null) => void;
  selectedKeyAlias: string | null;
  setSelectedKeyAlias: Setter<string | null>;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  organizations: Organization[] | null;
  setCurrentOrg: React.Dispatch<React.SetStateAction<Organization | null>>;
  refresh?: () => void;
  onSortChange?: (sortBy: string, sortOrder: "asc" | "desc") => void;
  currentSort?: {
    sortBy: string;
    sortOrder: "asc" | "desc";
  };
  premiumUser: boolean;
  setAccessToken?: (token: string) => void;
}

// Define columns similar to our logs table

interface UserResponse {
  user_id: string;
  user_email: string;
  user_role: string;
}

const TeamFilter = ({
  teams,
  selectedTeam,
  setSelectedTeam,
}: {
  teams: Team[] | null;
  selectedTeam: Team | null;
  setSelectedTeam: (team: Team | null) => void;
}) => {
  const handleTeamChange = (value: string) => {
    const team = teams?.find((t) => t.team_id === value);
    setSelectedTeam(team || null);
  };

  return (
    <div className="mb-4">
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-600">Where Team is</span>
        <Select
          value={selectedTeam?.team_id || ""}
          onValueChange={handleTeamChange}
          placeholder="Team ID"
          className="w-[400px]"
        >
          <SelectItem value="team_id">Team ID</SelectItem>
          {teams?.map((team) => (
            <SelectItem key={team.team_id} value={team.team_id}>
              <span className="font-medium">{team.team_alias}</span>{" "}
              <span className="text-gray-500">({team.team_id})</span>
            </SelectItem>
          ))}
        </Select>
      </div>
    </div>
  );
};

/**
 * AllKeysTable – a new table for keys that mimics the table styling used in view_logs.
 * The team selector and filtering have been removed so that all keys are shown.
 */

export function AllKeysTable({
  keys,
  setKeys,
  isLoading = false,
  pagination,
  onPageChange,
  pageSize = 50,
  teams,
  selectedTeam,
  setSelectedTeam,
  selectedKeyAlias,
  setSelectedKeyAlias,
  accessToken,
  userID,
  userRole,
  organizations,
  setCurrentOrg,
  refresh,
  onSortChange,
  currentSort,
  premiumUser,
  setAccessToken,
}: AllKeysTableProps) {
  const [selectedKeyId, setSelectedKeyId] = useState<string | null>(null);
  const [userList, setUserList] = useState<UserResponse[]>([]);
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
  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});

  // Use the filter logic hook

  const { filters, filteredKeys, allKeyAliases, allTeams, allOrganizations, handleFilterChange, handleFilterReset } =
    useFilterLogic({
      keys,
      teams,
      organizations,
      accessToken,
    });

  useEffect(() => {
    if (accessToken) {
      const user_IDs = keys.map((key) => key.user_id).filter((id) => id !== null);
      const fetchUserList = async () => {
        const userListData = await userListCall(accessToken, user_IDs, 1, 100);
        setUserList(userListData.users);
      };
      fetchUserList();
    }
  }, [accessToken, keys]);

  // Add a useEffect to call refresh when a key is created
  useEffect(() => {
    if (refresh) {
      const handleStorageChange = () => {
        refresh();
      };

      // Listen for storage events that might indicate a key was created
      window.addEventListener("storage", handleStorageChange);

      return () => {
        window.removeEventListener("storage", handleStorageChange);
      };
    }
  }, [refresh]);

  const columns: ColumnDef<KeyResponse>[] = [
    {
      id: "expander",
      header: () => null,
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
      cell: (info) => (
        <div className="overflow-hidden">
          <Tooltip title={info.getValue() as string}>
            <Button
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
              onClick={() => setSelectedKeyId(info.getValue() as string)}
            >
              {info.getValue() ? `${(info.getValue() as string).slice(0, 7)}...` : "-"}
            </Button>
          </Tooltip>
        </div>
      ),
    },
    {
      id: "key_alias",
      accessorKey: "key_alias",
      header: "Key Alias",
      cell: (info) => {
        const value = info.getValue() as string;
        return (
          <Tooltip title={value}>{value ? (value.length > 20 ? `${value.slice(0, 20)}...` : value) : "-"}</Tooltip>
        );
      },
    },
    {
      id: "key_name",
      accessorKey: "key_name",
      header: "Secret Key",
      cell: (info) => <span className="font-mono text-xs">{info.getValue() as string}</span>,
    },
    {
      id: "team_alias",
      accessorKey: "team_id",
      header: "Team Alias",
      cell: ({ row, getValue }) => {
        const teamId = getValue() as string;
        const team = teams?.find((t) => t.team_id === teamId);
        return team?.team_alias || "Unknown";
      },
    },
    {
      id: "team_id",
      accessorKey: "team_id",
      header: "Team ID",
      cell: (info) => (
        <Tooltip title={info.getValue() as string}>
          {info.getValue() ? `${(info.getValue() as string).slice(0, 7)}...` : "-"}
        </Tooltip>
      ),
    },
    {
      id: "organization_id",
      accessorKey: "organization_id",
      header: "Organization ID",
      cell: (info) => (info.getValue() ? info.renderValue() : "-"),
    },
    {
      id: "user_email",
      accessorKey: "user_id",
      header: "User Email",
      cell: (info) => {
        const userId = info.getValue() as string;
        const user = userList.find((u) => u.user_id === userId);
        return user?.user_email ? (
          <Tooltip title={user?.user_email}>
            <span>{user?.user_email.slice(0, 20)}...</span>
          </Tooltip>
        ) : (
          "-"
        );
      },
    },
    {
      id: "user_id",
      accessorKey: "user_id",
      header: "User ID",
      cell: (info) => {
        const userId = info.getValue() as string | null;
        if (userId && userId.length > 15) {
          return (
            <Tooltip title={userId}>
              <span>{userId.slice(0, 7)}...</span>
            </Tooltip>
          );
        }
        return userId ? userId : "-";
      },
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      header: "Created At",
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "-";
      },
    },
    {
      id: "created_by",
      accessorKey: "created_by",
      header: "Created By",
      cell: (info) => {
        const value = info.getValue() as string | null;
        if (value && value.length > 15) {
          return (
            <Tooltip title={value}>
              <span>{value.slice(0, 7)}...</span>
            </Tooltip>
          );
        }
        return value;
      },
    },
    {
      id: "updated_at",
      accessorKey: "updated_at",
      header: "Updated At",
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "Never";
      },
    },
    {
      id: "expires",
      accessorKey: "expires",
      header: "Expires",
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "Never";
      },
    },
    {
      id: "spend",
      accessorKey: "spend",
      header: "Spend (USD)",
      cell: (info) => formatNumberWithCommas(info.getValue() as number, 4),
    },
    {
      id: "max_budget",
      accessorKey: "max_budget",
      header: "Budget (USD)",
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
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleString() : "Never";
      },
    },
    {
      id: "models",
      accessorKey: "models",
      header: "Models",
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
  ];

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
      isSearchable: true,
      searchFn: async (searchText) => {
        const filteredKeyAliases = allKeyAliases.filter((key) => {
          return key.toLowerCase().includes(searchText.toLowerCase());
        });

        return filteredKeyAliases.map((key) => {
          return {
            label: key,
            value: key,
          };
        });
      },
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

  console.log(`keys: ${JSON.stringify(keys)}`);

  const table = useReactTable({
    data: filteredKeys,
    columns: columns.filter((col) => col.id !== "expander"),
    state: {
      sorting,
    },
    onSortingChange: (updaterOrValue) => {
      const newSorting = typeof updaterOrValue === "function" ? updaterOrValue(sorting) : updaterOrValue;
      console.log(`newSorting: ${JSON.stringify(newSorting)}`);
      setSorting(newSorting);
      if (newSorting && newSorting.length > 0) {
        const sortState = newSorting[0];
        const sortBy = sortState.id;
        const sortOrder = sortState.desc ? "desc" : "asc";
        console.log(`sortBy: ${sortBy}, sortOrder: ${sortOrder}`);
        handleFilterChange({
          ...filters,
          "Sort By": sortBy,
          "Sort Order": sortOrder,
        });
        onSortChange?.(sortBy, sortOrder);
      }
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
    manualSorting: false,
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

  return (
    <div className="w-full h-full overflow-hidden">
      {selectedKeyId ? (
        <KeyInfoView
          keyId={selectedKeyId}
          onClose={() => setSelectedKeyId(null)}
          keyData={filteredKeys.find((k) => k.token === selectedKeyId)}
          onKeyDataUpdate={(updatedKeyData) => {
            setKeys((keys) =>
              keys.map((key) => {
                if (key.token === updatedKeyData.token) {
                  return updateExistingKeys(key, updatedKeyData);
                }
                return key;
              }),
            );
            if (refresh) refresh(); // Minimal fix: refresh the full key list after an update
          }}
          onDelete={() => {
            setKeys((keys) => keys.filter((key) => key.token !== selectedKeyId));
            if (refresh) refresh(); // Minimal fix: refresh the full key list after a delete
          }}
          accessToken={accessToken}
          userID={userID}
          userRole={userRole}
          teams={allTeams}
          premiumUser={premiumUser}
          setAccessToken={setAccessToken}
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
            <span className="inline-flex text-sm text-gray-700">
              Showing{" "}
              {isLoading
                ? "..."
                : `${(pagination.currentPage - 1) * pageSize + 1} - ${Math.min(pagination.currentPage * pageSize, pagination.totalCount)}`}{" "}
              of {isLoading ? "..." : pagination.totalCount} results
            </span>

            <div className="inline-flex items-center gap-2">
              <span className="text-sm text-gray-700">
                Page {isLoading ? "..." : pagination.currentPage} of {isLoading ? "..." : pagination.totalPages}
              </span>

              <button
                onClick={() => onPageChange(pagination.currentPage - 1)}
                disabled={isLoading || pagination.currentPage === 1}
                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>

              <button
                onClick={() => onPageChange(pagination.currentPage + 1)}
                disabled={isLoading || pagination.currentPage === pagination.totalPages}
                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
          <div className="h-[75vh] overflow-auto">
            <div className="rounded-lg custom-border relative">
              <div className="overflow-x-auto">
                <Table className="[&_td]:py-0.5 [&_th]:py-1">
                  <TableHead>
                    {table.getHeaderGroups().map((headerGroup) => (
                      <TableRow key={headerGroup.id}>
                        {headerGroup.headers.map((header) => (
                          <TableHeaderCell
                            key={header.id}
                            className={`py-1 h-8 ${
                              header.id === "actions"
                                ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                                : ""
                            }`}
                            onClick={header.column.getToggleSortingHandler()}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center">
                                {header.isPlaceholder
                                  ? null
                                  : flexRender(header.column.columnDef.header, header.getContext())}
                              </div>
                              {header.id !== "actions" && (
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
                                maxWidth: "8-x",
                                whiteSpace: "pre-wrap",
                                overflow: "hidden",
                              }}
                              className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${cell.column.id === "models" && (cell.getValue() as string[]).length > 3 ? "px-0" : ""}`}
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
