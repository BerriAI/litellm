import { CopyOutlined } from "@ant-design/icons";
import {
  ChevronDownIcon,
  ChevronUpIcon,
  SwitchVerticalIcon,
  TrashIcon,
} from "@heroicons/react/outline";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  Badge,
  Button,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { Switch, Tooltip } from "antd";
import React, { useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import {
  disableClaudeCodePlugin,
  enableClaudeCodePlugin,
} from "../networking";
import {
  getCategoryBadgeColor
} from "./helpers";
import { Plugin } from "./types";

interface PluginTableProps {
  pluginsList: Plugin[];
  isLoading: boolean;
  onDeleteClick: (pluginName: string, displayName: string) => void;
  accessToken: string | null;
  onPluginUpdated: () => void;
  isAdmin: boolean;
  onPluginClick: (pluginId: string) => void;
}

const PluginTable: React.FC<PluginTableProps> = ({
  pluginsList,
  isLoading,
  onDeleteClick,
  accessToken,
  onPluginUpdated,
  isAdmin,
  onPluginClick,
}) => {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);
  const [togglingPlugin, setTogglingPlugin] = useState<string | null>(null);

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Copied to clipboard!");
  };

  const handleToggleEnabled = async (plugin: Plugin) => {
    if (!accessToken) return;

    setTogglingPlugin(plugin.id);
    try {
      if (plugin.enabled) {
        await disableClaudeCodePlugin(accessToken, plugin.name);
        NotificationsManager.success(`Plugin "${plugin.name}" disabled`);
      } else {
        await enableClaudeCodePlugin(accessToken, plugin.name);
        NotificationsManager.success(`Plugin "${plugin.name}" enabled`);
      }
      onPluginUpdated();
    } catch (error) {
      NotificationsManager.error("Failed to toggle plugin status");
    } finally {
      setTogglingPlugin(null);
    }
  };

  const columns: ColumnDef<Plugin>[] = [
    {
      header: "Plugin Name",
      accessorKey: "name",
      cell: ({ row }) => {
        const plugin = row.original;
        const name = plugin.name || "";
        return (
          <div className="flex items-center gap-2">
            <Tooltip title={name}>
              <Button
                size="xs"
                variant="light"
                className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate min-w-[150px] justify-start"
                onClick={() => onPluginClick(plugin.id)}
              >
                {name}
              </Button>
            </Tooltip>
            <Tooltip title="Copy Plugin ID">
              <CopyOutlined
                onClick={(e) => {
                  e.stopPropagation();
                  copyToClipboard(plugin.id);
                }}
                className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
              />
            </Tooltip>
          </div>
        );
      },
    },
    {
      header: "Version",
      accessorKey: "version",
      cell: ({ row }) => {
        const version = row.original.version || "N/A";
        return <span className="text-xs text-gray-600">{version}</span>;
      },
    },
    {
      header: "Description",
      accessorKey: "description",
      cell: ({ row }) => {
        const description = row.original.description || "No description";
        return (
          <Tooltip title={description}>
            <span className="text-xs text-gray-600 block max-w-[300px] truncate">
              {description}
            </span>
          </Tooltip>
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
            <Badge color="gray" className="text-xs font-normal" size="xs">
              Uncategorized
            </Badge>
          );
        }
        const badgeColor = getCategoryBadgeColor(category);
        return (
          <Badge color={badgeColor} className="text-xs font-normal" size="xs">
            {category}
          </Badge>
        );
      },
    },
    {
      header: "Enabled",
      accessorKey: "enabled",
      cell: ({ row }) => {
        const plugin = row.original;
        return (
          <div className="flex items-center gap-2">
            <Badge
              color={plugin.enabled ? "green" : "gray"}
              className="text-xs font-normal"
              size="xs"
            >
              {plugin.enabled ? "Yes" : "No"}
            </Badge>
            {isAdmin && (
              <Tooltip
                title={plugin.enabled ? "Disable plugin" : "Enable plugin"}
              >
                <Switch
                  size="small"
                  checked={plugin.enabled}
                  loading={togglingPlugin === plugin.id}
                  onChange={() => handleToggleEnabled(plugin)}
                />
              </Tooltip>
            )}
          </div>
        );
      },
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      cell: ({ row }) => {
        const plugin = row.original;
        return (
          <Tooltip title={plugin.created_at}>
            <span className="text-xs">{formatDate(plugin.created_at)}</span>
          </Tooltip>
        );
      },
    },
    ...(isAdmin
      ? [
        {
          header: "Actions",
          id: "actions",
          enableSorting: false,
          cell: ({ row }: any) => {
            const plugin = row.original;

            return (
              <div className="flex items-center gap-1">
                <Tooltip title="Delete plugin">
                  <Button
                    size="xs"
                    variant="light"
                    color="red"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteClick(plugin.name, plugin.name);
                    }}
                    icon={TrashIcon}
                    className="text-red-500 hover:text-red-700 hover:bg-red-50"
                  />
                </Tooltip>
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
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-0.5 [&_th]:py-1">
          <TableHead>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHeaderCell
                    key={header.id}
                    className={`py-1 h-8 ${header.id === "actions"
                        ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
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
                            header.getContext()
                          )}
                      </div>
                      {header.column.getCanSort() && (
                        <div className="w-4">
                          {header.column.getIsSorted() ? (
                            {
                              asc: (
                                <ChevronUpIcon className="h-4 w-4 text-blue-500" />
                              ),
                              desc: (
                                <ChevronDownIcon className="h-4 w-4 text-blue-500" />
                              ),
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
                    <p>Loading...</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : pluginsList && pluginsList.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${cell.column.id === "actions"
                          ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                          : ""
                        }`}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
                    <p>No plugins found. Add one to get started.</p>
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
