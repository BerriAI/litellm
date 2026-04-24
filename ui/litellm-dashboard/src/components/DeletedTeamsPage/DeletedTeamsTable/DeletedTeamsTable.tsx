"use client";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ArrowUpDown, ChevronDown, ChevronUp } from "lucide-react";
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
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
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
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="truncate block max-w-[200px]">
                  {value || "-"}
                </span>
              </TooltipTrigger>
              {value && <TooltipContent>{value}</TooltipContent>}
            </Tooltip>
          </TooltipProvider>
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
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="font-mono text-blue-500 text-xs truncate block max-w-[250px]">
                  {value || "-"}
                </span>
              </TooltipTrigger>
              <TooltipContent>{value}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
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
        const spend = (info.row.original as { spend?: number }).spend;
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
        const redBadge =
          "text-xs bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300";
        const blueBadge =
          "text-xs bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300";
        const grayBadge = "text-xs bg-muted text-muted-foreground";
        if (!Array.isArray(models) || models.length === 0) {
          return <Badge className={redBadge}>All Proxy Models</Badge>;
        }
        return (
          <div className="flex flex-wrap gap-1 max-w-[300px]">
            {models.slice(0, 3).map((model: string, index: number) =>
              model === "all-proxy-models" ? (
                <Badge key={index} className={redBadge}>
                  All Proxy Models
                </Badge>
              ) : (
                <Badge key={index} className={blueBadge}>
                  {model.length > 30
                    ? `${getModelDisplayName(model).slice(0, 30)}...`
                    : getModelDisplayName(model)}
                </Badge>
              ),
            )}
            {models.length > 3 && (
              <Badge className={grayBadge}>
                +{models.length - 3}{" "}
                {models.length - 3 === 1 ? "more model" : "more models"}
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
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="truncate block max-w-[200px]">
                  {value || "-"}
                </span>
              </TooltipTrigger>
              {value && <TooltipContent>{value}</TooltipContent>}
            </Tooltip>
          </TooltipProvider>
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
        const value = (info.row.original as { deleted_at?: string | null })
          .deleted_at;
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
        const value = (info.row.original as { deleted_by?: string | null })
          .deleted_by;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="truncate block max-w-[180px]">
                  {value || "-"}
                </span>
              </TooltipTrigger>
              {value && <TooltipContent>{value}</TooltipContent>}
            </Tooltip>
          </TooltipProvider>
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
            <span className="inline-flex text-sm text-foreground">
              Loading...
            </span>
          ) : (
            <span className="inline-flex text-sm text-foreground">
              Showing {teams.length} {teams.length === 1 ? "team" : "teams"}
            </span>
          )}
        </div>
        <div className="h-[75vh] overflow-auto">
          <div className="rounded-lg custom-border relative">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1" style={{ width: table.getCenterTotalSize() }}>
                <TableHeader>
                  {table.getHeaderGroups().map((headerGroup) => (
                    <TableRow key={headerGroup.id}>
                      {headerGroup.headers.map((header) => (
                        <TableHead
                          key={header.id}
                          data-header-id={header.id}
                          className={cn("py-1 h-8 relative hover:bg-muted")}
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
                                  asc: (
                                    <ChevronUp className="h-4 w-4 text-primary" />
                                  ),
                                  desc: (
                                    <ChevronDown className="h-4 w-4 text-primary" />
                                  ),
                                }[header.column.getIsSorted() as string]
                              ) : (
                                <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
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
                        </TableHead>
                      ))}
                    </TableRow>
                  ))}
                </TableHeader>
                <TableBody>
                  {isLoading || isFetching ? (
                    <TableRow>
                      <TableCell
                        colSpan={columns.length}
                        className="h-8 text-center"
                      >
                        <div className="text-center text-muted-foreground">
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
                      <TableCell
                        colSpan={columns.length}
                        className="h-8 text-center"
                      >
                        <div className="text-center text-muted-foreground">
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
