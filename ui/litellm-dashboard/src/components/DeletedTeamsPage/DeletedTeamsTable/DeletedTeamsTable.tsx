"use client";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ChevronDownIcon, ChevronUpIcon, SwitchVerticalIcon } from "@heroicons/react/outline";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Badge,
  Text,
} from "@tremor/react";
import { Tooltip } from "antd";
import React, { useState } from "react";
import { DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";

interface DeletedTeamsTableProps {
  teams: DeletedTeam[];
  isLoading: boolean;
  isFetching: boolean;
}

export function DeletedTeamsTable({
  teams,
  isLoading,
  isFetching,
}: DeletedTeamsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([
    {
      id: "deleted_at",
      desc: true,
    },
  ]);

  const columns: ColumnDef<DeletedTeam>[] = [
    {
      id: "team_alias",
      accessorKey: "team_alias",
      header: "Team Name",
      size: 150,
      maxSize: 200,
      cell: (info) => {
        const value = info.getValue() as string;
        return (
          <Tooltip title={value || undefined}>
            <span className="truncate block max-w-[200px]">
              {value || "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      id: "team_id",
      accessorKey: "team_id",
      header: "Team ID",
      size: 150,
      maxSize: 250,
      cell: (info) => {
        const value = info.getValue() as string;
        return (
          <Tooltip title={value}>
            <span className="font-mono text-blue-500 text-xs truncate block max-w-[250px]">
              {value || "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      header: "Created",
      size: 120,
      maxSize: 140,
      cell: (info) => {
        const value = info.getValue();
        return (
          <span className="block max-w-[140px]">
            {value ? new Date(value as string).toLocaleDateString() : "-"}
          </span>
        );
      },
    },
    {
      id: "spend",
      accessorKey: "spend",
      header: "Spend (USD)",
      size: 100,
      maxSize: 140,
      cell: (info) => {
        const spend = (info.row.original as any).spend as number | undefined;
        return (
          <span className="block max-w-[140px]">
            {spend !== undefined ? formatNumberWithCommas(spend, 4) : "-"}
          </span>
        );
      },
    },
    {
      id: "max_budget",
      accessorKey: "max_budget",
      header: "Budget (USD)",
      size: 110,
      maxSize: 150,
      cell: (info) => {
        const maxBudget = info.getValue() as number | null;
        return (
          <span className="block max-w-[150px]">
            {maxBudget === null || maxBudget === undefined ? "No limit" : `$${formatNumberWithCommas(maxBudget)}`}
          </span>
        );
      },
    },
    {
      id: "models",
      accessorKey: "models",
      header: "Models",
      size: 200,
      maxSize: 300,
      cell: (info) => {
        const models = info.getValue() as string[];
        if (!Array.isArray(models) || models.length === 0) {
          return (
            <Badge size={"xs"} color="red">
              <Text>All Proxy Models</Text>
            </Badge>
          );
        }
        return (
          <div className="flex flex-wrap gap-1 max-w-[300px]">
            {models.slice(0, 3).map((model: string, index: number) =>
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
            {models.length > 3 && (
              <Badge size={"xs"} color="gray">
                <Text>
                  +{models.length - 3} {models.length - 3 === 1 ? "more model" : "more models"}
                </Text>
              </Badge>
            )}
          </div>
        );
      },
    },
    {
      id: "organization_id",
      accessorKey: "organization_id",
      header: "Organization",
      size: 150,
      maxSize: 200,
      cell: (info) => {
        const value = info.getValue() as string;
        return (
          <Tooltip title={value || undefined}>
            <span className="truncate block max-w-[200px]">
              {value || "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      id: "deleted_at",
      accessorKey: "deleted_at",
      header: "Deleted At",
      size: 120,
      maxSize: 140,
      cell: (info) => {
        const value = (info.row.original as any).deleted_at as string | null | undefined;
        return (
          <span className="block max-w-[140px]">
            {value ? new Date(value).toLocaleDateString() : "-"}
          </span>
        );
      },
    },
    {
      id: "deleted_by",
      accessorKey: "deleted_by",
      header: "Deleted By",
      size: 120,
      maxSize: 180,
      cell: (info) => {
        const value = (info.row.original as any).deleted_by as string | null | undefined;
        return (
          <Tooltip title={value || undefined}>
            <span className="truncate block max-w-[180px]">
              {value || "-"}
            </span>
          </Tooltip>
        );
      },
    },
  ];

  const table = useReactTable({
    data: teams,
    columns,
    columnResizeMode: "onChange",
    columnResizeDirection: "ltr",
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
    manualSorting: false,
  });

  return (
    <div className="w-full h-full overflow-hidden">
      <div className="border-b py-4 flex-1 overflow-hidden">
        <div className="flex items-center justify-between w-full mb-4">
          {isLoading || isFetching ? (
            <span className="inline-flex text-sm text-gray-700">Loading...</span>
          ) : (
            <span className="inline-flex text-sm text-gray-700">
              Showing {teams.length} {teams.length === 1 ? "team" : "teams"}
            </span>
          )}
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
                          className={`py-1 h-8 relative hover:bg-gray-50`}
                          style={{
                            width: header.getSize(),
                            maxWidth: header.column.columnDef.maxSize,
                            position: "relative",
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
                          onClick={header.column.getToggleSortingHandler()}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center">
                              {header.isPlaceholder
                                ? null
                                : flexRender(header.column.columnDef.header, header.getContext())}
                            </div>
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
                  {isLoading || isFetching ? (
                    <TableRow>
                      <TableCell colSpan={columns.length} className="h-8 text-center">
                        <div className="text-center text-gray-500">
                          <p>🚅 Loading teams...</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : teams.length > 0 ? (
                    table.getRowModel().rows.map((row) => (
                      <TableRow key={row.id} className="h-8">
                        {row.getVisibleCells().map((cell) => (
                          <TableCell
                            key={cell.id}
                            style={{
                              width: cell.column.getSize(),
                              maxWidth: cell.column.columnDef.maxSize,
                              whiteSpace: "pre-wrap",
                              overflow: "hidden",
                            }}
                            className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap"
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
                          <p>No deleted teams found</p>
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
    </div>
  );
}
