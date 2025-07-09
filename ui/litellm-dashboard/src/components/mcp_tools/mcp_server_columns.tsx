import { ColumnDef } from "@tanstack/react-table";
import { MCPServer, MCPAccessGroup } from "./types";
import { Icon } from "@tremor/react";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { getMaskedAndFullUrl } from "./utils";

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
    accessorKey: "mcp_access_groups",
    header: "Access Groups",
    cell: ({ row }) => {
      const groups = row.original.mcp_access_groups || [];
      return (
        <div className="flex flex-wrap gap-1">
          {groups.map((group: MCPAccessGroup) => (
            <span
              key={group.group_id}
              className="bg-blue-50 text-blue-700 text-xs px-2 py-0.5 rounded flex items-center gap-1"
              title={group.description || ''}
            >
              {group.group_name}
            </span>
          ))}
        </div>
      );
    },
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
    id: "actions",
    header: "Info",
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
