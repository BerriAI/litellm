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
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Copy,
  Trash2,
} from "lucide-react";
import React, { useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import { getCategoryBadgeColor } from "./helpers";
import { Plugin } from "./types";

interface PluginTableProps {
  pluginsList: Plugin[];
  isLoading: boolean;
  onDeleteClick: (pluginName: string, displayName: string) => void;
  accessToken: string | null;
  isAdmin: boolean;
  onPluginClick: (pluginId: string) => void;
}

// Map tremor-style badge color tokens to Tailwind palette classes.
const BADGE_COLOR_CLASSES: Record<string, string> = {
  gray: "bg-muted text-muted-foreground",
  green:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  red: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  blue: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  indigo: "bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300",
  purple: "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  orange: "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
  amber: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  yellow: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  cyan: "bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300",
  pink: "bg-pink-100 text-pink-700 dark:bg-pink-950 dark:text-pink-300",
  rose: "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300",
  teal: "bg-teal-100 text-teal-700 dark:bg-teal-950 dark:text-teal-300",
};

const badgeClasses = (color: string): string =>
  BADGE_COLOR_CLASSES[color] || BADGE_COLOR_CLASSES.gray;

const PluginTable: React.FC<PluginTableProps> = ({
  pluginsList,
  isLoading,
  onDeleteClick,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  accessToken,
  isAdmin,
  onPluginClick,
}) => {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Copied to clipboard!");
  };

  const columns: ColumnDef<Plugin>[] = [
    {
      header: "Skill Name",
      accessorKey: "name",
      cell: ({ row }) => {
        const plugin = row.original;
        const name = plugin.name || "";
        return (
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => onPluginClick(plugin.id)}
                    className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate min-w-[150px] rounded"
                  >
                    {name}
                  </button>
                </TooltipTrigger>
                <TooltipContent>{name}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      copyToClipboard(plugin.id);
                    }}
                    className="cursor-pointer text-muted-foreground hover:text-primary"
                    aria-label="Copy Plugin ID"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Copy Plugin ID</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        );
      },
    },
    {
      header: "Version",
      accessorKey: "version",
      cell: ({ row }) => (
        <span className="text-xs text-muted-foreground">
          {row.original.version || "N/A"}
        </span>
      ),
    },
    {
      header: "Description",
      accessorKey: "description",
      cell: ({ row }) => {
        const description = row.original.description || "No description";
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs text-muted-foreground block max-w-[300px] truncate">
                  {description}
                </span>
              </TooltipTrigger>
              <TooltipContent>{description}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Category",
      accessorKey: "category",
      cell: ({ row }) => {
        const category = row.original.category;
        if (!category) {
          return (
            <Badge className={cn("text-xs font-normal", badgeClasses("gray"))}>
              Uncategorized
            </Badge>
          );
        }
        const badgeColor = getCategoryBadgeColor(category);
        return (
          <Badge className={cn("text-xs font-normal", badgeClasses(badgeColor))}>
            {category}
          </Badge>
        );
      },
    },
    {
      header: "Public",
      accessorKey: "enabled",
      cell: ({ row }) => {
        const plugin = row.original;
        return (
          <Badge
            className={cn(
              "text-xs font-normal",
              badgeClasses(plugin.enabled ? "green" : "gray"),
            )}
          >
            {plugin.enabled ? "Yes" : "No"}
          </Badge>
        );
      },
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      cell: ({ row }) => {
        const plugin = row.original;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">{formatDate(plugin.created_at)}</span>
              </TooltipTrigger>
              <TooltipContent>{plugin.created_at}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    ...(isAdmin
      ? [
          {
            header: "Actions",
            id: "actions",
            enableSorting: false,
            cell: ({ row }: { row: { original: Plugin } }) => {
              const plugin = row.original;
              return (
                <div className="flex items-center gap-1">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteClick(plugin.name, plugin.name);
                          }}
                          aria-label="Delete skill"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Delete skill</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              );
            },
          },
        ]
      : []),
  ];

  const table = useReactTable({
    data: pluginsList,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-0.5 [&_th]:py-1">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    className={`py-1 h-8 ${
                      header.id === "actions"
                        ? "sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                        : ""
                    }`}
                    onClick={
                      header.column.getCanSort()
                        ? header.column.getToggleSortingHandler()
                        : undefined
                    }
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext(),
                            )}
                      </div>
                      {header.column.getCanSort() && (
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
                      )}
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-8 text-center"
                >
                  <div className="text-center text-muted-foreground">
                    <p>Loading...</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : pluginsList && pluginsList.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  className="h-8 cursor-pointer hover:bg-muted"
                  onClick={() => onPluginClick(row.original.id)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${
                        cell.column.id === "actions"
                          ? "sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                          : ""
                      }`}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
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
                    <p>No skills found. Add one to get started.</p>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

export default PluginTable;
