import { RefreshIcon, TrashIcon } from "@heroicons/react/outline";
import { ColumnDef, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { Button, Tooltip } from "antd";
import React from "react";
import { DateCell, StatusBadge, StatusTone } from "@/components/shared/table_cells";
import { MarketplaceSource } from "@/components/claude_code_plugins/types";

interface MarketplaceTableProps {
  marketplacesList: MarketplaceSource[];
  isLoading: boolean;
  isAdmin: boolean;
  syncingName: string | null;
  onSyncClick: (marketplaceName: string) => void;
  onDeleteClick: (marketplaceName: string, displayName: string) => void;
}

const SYNC_STATUS_TONE: Record<string, StatusTone> = {
  synced: "success",
  syncing: "info",
  pending: "warning",
  error: "error",
};

const syncStatusTone = (status: string): StatusTone => SYNC_STATUS_TONE[status] ?? "neutral";

const MarketplaceTable: React.FC<MarketplaceTableProps> = ({
  marketplacesList,
  isLoading,
  isAdmin,
  syncingName,
  onSyncClick,
  onDeleteClick,
}) => {
  const columns: ColumnDef<MarketplaceSource>[] = [
    {
      header: "Marketplace",
      accessorKey: "name",
      cell: ({ row }) => {
        const marketplace = row.original;
        return (
          <div className="flex flex-col">
            <span className="font-mono text-xs text-gray-900">{marketplace.name}</span>
            {marketplace.display_name && marketplace.display_name !== marketplace.name && (
              <span className="text-xs text-gray-500">{marketplace.display_name}</span>
            )}
          </div>
        );
      },
    },
    {
      header: "Source",
      accessorKey: "source_type",
      cell: ({ row }) => {
        const marketplace = row.original;
        return (
          <Tooltip title={marketplace.source_ref}>
            <span className="text-xs text-gray-600">{marketplace.source_type}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Sync Status",
      accessorKey: "sync_status",
      cell: ({ row }) => {
        const marketplace = row.original;
        return (
          <StatusBadge
            tone={syncStatusTone(marketplace.sync_status)}
            label={marketplace.sync_status}
            tooltip={marketplace.sync_error || undefined}
          />
        );
      },
    },
    {
      header: "Skills",
      accessorKey: "plugin_count",
      cell: ({ row }) => <span className="text-xs text-gray-600">{row.original.plugin_count}</span>,
    },
    {
      header: "Last Synced",
      accessorKey: "last_synced_at",
      cell: ({ row }) => <DateCell value={row.original.last_synced_at} />,
    },
    ...(isAdmin
      ? [
          {
            header: "Actions",
            id: "actions",
            cell: ({ row }: { row: { original: MarketplaceSource } }) => {
              const marketplace = row.original;
              return (
                <div className="flex items-center gap-1">
                  <Tooltip title="Sync now">
                    <Button
                      size="small"
                      type="text"
                      onClick={() => onSyncClick(marketplace.name)}
                      icon={<RefreshIcon className="h-4 w-4" />}
                      loading={syncingName === marketplace.name}
                      disabled={syncingName !== null}
                    />
                  </Tooltip>
                  <Tooltip title="Delete marketplace">
                    <Button
                      size="small"
                      type="text"
                      onClick={() => onDeleteClick(marketplace.name, marketplace.display_name || marketplace.name)}
                      icon={<TrashIcon className="h-4 w-4" />}
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
    data: marketplacesList,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <table className="min-w-full [&_td]:py-0.5 [&_th]:py-1">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th key={header.id} className="py-1 h-8 px-4 text-left text-xs font-medium text-gray-500">
                    {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {isLoading && (
              <tr>
                <td colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
                    <p>Loading...</p>
                  </div>
                </td>
              </tr>
            )}
            {!isLoading && marketplacesList.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
                    <p>No marketplaces imported. Add one to get started.</p>
                  </div>
                </td>
              </tr>
            )}
            {!isLoading &&
              marketplacesList.length > 0 &&
              table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="h-8 border-t border-gray-100">
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="py-0.5 px-4 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MarketplaceTable;
