import { ColumnDef } from "@tanstack/react-table";
import { MCPServer } from "./types";
import { Icon } from "@tremor/react";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { getMaskedAndFullUrl } from "./utils";
import { Tooltip } from "antd";

export const mcpServerColumns = (
  userRole: string,
  onView: (serverId: string) => void,
  onEdit: (serverId: string) => void,
  onDelete: (serverId: string) => void
): ColumnDef<MCPServer>[] => [
  {
    accessorKey: "server_id",
    header: "Server ID",
    cell: ({ row }) => (
      <button
        onClick={() => onView(row.original.server_id)}
        className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch]"
      >
        {row.original.server_id.slice(0, 7)}...
      </button>
    ),
  },
  {
    accessorKey: "alias",
    header: "Name",
  },
  {
    id: "url",
    header: "URL",
    cell: ({ row }) => {
      const { maskedUrl } = getMaskedAndFullUrl(row.original.url);
      return (
        <span className="font-mono text-sm">
          {maskedUrl}
        </span>
      );
    },
  },
  {
    accessorKey: "transport",
    header: "Transport",
    cell: ({ getValue }) => (
      <span>
        {(getValue() as string || "http").toUpperCase()}
      </span>
    ),
  },
  {
    accessorKey: "auth_type",
    header: "Auth Type",
    cell: ({ getValue }) => (
      <span>
        {getValue() as string || "none"}
      </span>
    ),
  },
  {
    id: "mcp_access_groups",
    header: "Access Groups",
    cell: ({ row }) => {
      const groups = row.original.mcp_access_groups;
      if (Array.isArray(groups) && groups.length > 0) {
        // If string array
        if (typeof groups[0] === "string") {
          const joined = groups.join(", ");
          return (
            <Tooltip title={joined}>
              <span className="max-w-[200px] truncate block">{joined.length > 30 ? `${joined.slice(0, 30)}...` : joined}</span>
            </Tooltip>
          );
        }
      }
      return <span className="text-gray-400 italic">None</span>;
    },
  },
  {
    header: "Created At",
    accessorKey: "created_at",
    sortingFn: "datetime",
    cell: ({ row }) => {
      const server = row.original;
      return (
        <span className="text-xs">
          {server.created_at ? new Date(server.created_at).toLocaleDateString() : "-"}
        </span>
      );
    },
  },
  {
    header: "Updated At",
    accessorKey: "updated_at",
    sortingFn: "datetime",
    cell: ({ row }) => {
      const server = row.original;
      return (
        <span className="text-xs">
          {server.updated_at ? new Date(server.updated_at).toLocaleDateString() : "-"}
        </span>
      );
    },
  },
  {
    id: "actions",
    header: "Actions",
    cell: ({ row }) => (
      <div className="flex items-center gap-2">
        <Icon
          icon={PencilAltIcon}
          size="sm"
          onClick={() => onEdit(row.original.server_id)}
          className="cursor-pointer"
        />
        <Icon
          icon={TrashIcon}
          size="sm"
          onClick={() => onDelete(row.original.server_id)}
          className="cursor-pointer"
        />
      </div>
    ),
  },
];
