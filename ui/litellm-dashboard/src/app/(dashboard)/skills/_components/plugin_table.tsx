import { CopyOutlined } from "@ant-design/icons";
import { TrashIcon } from "@heroicons/react/outline";
import { ColumnDef, PaginationState, SortingState } from "@tanstack/react-table";
import { Badge, Button } from "@tremor/react";
import { Tooltip } from "antd";
import React, { useState } from "react";
import { DataTable, DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell, StatusBadge } from "@/components/shared/table_cells";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { getCategoryBadgeColor } from "@/components/claude_code_plugins/helpers";
import { Plugin } from "@/components/claude_code_plugins/types";

interface PluginTableProps {
  pluginsList: Plugin[];
  isLoading: boolean;
  onDeleteClick: (pluginName: string, displayName: string) => void;
  accessToken: string | null;
  isAdmin: boolean;
  onPluginClick: (pluginId: string) => void;
}

const PluginTable: React.FC<PluginTableProps> = ({ pluginsList, isLoading, onDeleteClick, isAdmin, onPluginClick }) => {
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 25 });

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Copied to clipboard!");
  };

  const columns: ColumnDef<Plugin>[] = [
    {
      id: "name",
      header: ({ column }) => <DataTableSortHeader column={column} title="Skill Name" />,
      accessorKey: "name",
      cell: ({ row }) => {
        const plugin = row.original;
        return (
          <div className="flex items-center gap-2">
            <IdCell value={plugin.name} onClick={() => onPluginClick(plugin.id)} />
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
      id: "version",
      header: "Version",
      accessorKey: "version",
      enableSorting: false,
      cell: ({ row }) => {
        const version = row.original.version || "N/A";
        return <span className="text-xs text-gray-600">{version}</span>;
      },
    },
    {
      id: "description",
      header: "Description",
      accessorKey: "description",
      enableSorting: false,
      cell: ({ row }) => {
        const description = row.original.description || "No description";
        return (
          <Tooltip title={description}>
            <span className="text-xs text-gray-600 block max-w-[300px] truncate">{description}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "category",
      header: ({ column }) => <DataTableSortHeader column={column} title="Category" />,
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
      id: "enabled",
      header: ({ column }) => <DataTableSortHeader column={column} title="Public" />,
      accessorKey: "enabled",
      cell: ({ row }) => {
        const plugin = row.original;
        return (
          <StatusBadge
            tone={plugin.enabled ? "success" : "neutral"}
            label={plugin.enabled ? "Yes" : "No"}
            tooltip={plugin.enabled ? "Visible without a key" : "Requires an assigned grant"}
          />
        );
      },
    },
    {
      id: "created_at",
      header: ({ column }) => <DataTableSortHeader column={column} title="Created At" />,
      accessorKey: "created_at",
      cell: ({ row }) => <DateCell value={row.original.created_at} />,
    },
    ...(isAdmin
      ? [
          {
            id: "actions",
            header: "Actions",
            enableSorting: false,
            meta: { pinned: "right" as const },
            cell: ({ row }: { row: { original: Plugin } }) => {
              const plugin = row.original;

              return (
                <div className="flex items-center gap-1">
                  <Tooltip title="Delete skill">
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

  return (
    <DataTable
      data={pluginsList}
      columns={columns}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={setPagination}
      onRowClick={(plugin) => onPluginClick(plugin.id)}
      isLoading={isLoading}
      loadingMessage="Loading skills..."
      noDataMessage="No skills found. Add one to get started."
      size="compact"
    />
  );
};

export default PluginTable;
