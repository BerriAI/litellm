import React from "react";
import { ColumnDef } from "@tanstack/react-table";
import { Tooltip } from "antd";
import { Icon } from "@tremor/react";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { MCPServer, handleAuth, handleTransport } from "./types";
import { isAdminRole } from "@/utils/roles";
import { maskUrl } from "./utils";

const displayFriendlyId = (id: string) => `${id.slice(0, 7)}...`;

const displayFriendlyUrl = (url: string) => {
  if (!url) return "";
  return url.length > 30 ? `${url.slice(0, 30)}...` : url;
};

export const mcpServerColumns = (
  userRole: string | null,
  onSelect: (serverId: string) => void,
  onEdit: (serverId: string) => void,
  onDelete: (serverId: string) => void
): ColumnDef<MCPServer>[] => [
  {
    accessorKey: "server_id",
    header: "ID",
    cell: ({ getValue, row }) => (
      <button
        onClick={() => onSelect(row.original.server_id)}
        className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
      >
        {displayFriendlyId(getValue() as string)}
      </button>
    ),
  },
  {
    accessorKey: "alias",
    header: "Name",
  },
  {
    accessorKey: "url",
    header: "URL",
    cell: ({ getValue }) => (
      <Tooltip title={getValue() as string}>
        <span className="font-mono text-gray-600 text-xs">
          {maskUrl(getValue() as string)}
        </span>
      </Tooltip>
    ),
  },
  {
    accessorKey: "transport",
    header: "Transport",
    cell: ({ row }) => handleTransport(row.original.transport),
  },
  {
    accessorKey: "auth_type",
    header: "Auth Type",
    cell: ({ row }) => handleAuth(row.original.auth_type),
  },
  {
    id: "actions",
    header: "Info",
    cell: ({ row }) =>
      isAdminRole(userRole || "") ? (
        <>
          <Icon
            icon={PencilAltIcon}
            size="sm"
            onClick={() => onEdit(row.original.server_id)}
          />
          <Icon
            onClick={() => onDelete(row.original.server_id)}
            icon={TrashIcon}
            size="sm"
          />
        </>
      ) : null,
  },
];
