import { ColumnDef } from "@tanstack/react-table";
import { MCPServer } from "./types";
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
