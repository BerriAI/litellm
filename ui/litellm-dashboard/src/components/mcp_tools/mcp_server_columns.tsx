import React from "react";
import { ColumnDef } from "@tanstack/react-table";
import { Tooltip } from "antd";
import { Icon, Button } from "@tremor/react";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { MCPServer, handleAuth, handleTransport } from "./types";
import { isAdminRole } from "@/utils/roles";

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
    header: "Server ID",
    accessorKey: "server_id",
    cell: ({ row, getValue }) => (
      <div className="overflow-hidden">
        <Tooltip title={getValue() as string}>
          <Button
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
            onClick={() => onSelect(row.original.server_id)}
          >
            {displayFriendlyId(getValue() as string)}
          </Button>
        </Tooltip>
      </div>
    ),
  },
  {
    header: "Server Name",
    accessorKey: "alias",
    cell: ({ getValue }) => <span>{getValue() as string}</span>,
  },
  {
    header: "Description",
    accessorKey: "description",
    cell: ({ getValue }) => <span>{getValue() as string}</span>,
  },
  {
    header: "Transport",
    accessorKey: "transport",
    cell: ({ getValue }) => <span>{handleTransport(getValue() as string)}</span>,
  },
  {
    header: "Auth Type",
    accessorKey: "auth_type",
    cell: ({ getValue }) => <span>{handleAuth(getValue() as string)}</span>,
  },
  {
    header: "Url",
    accessorKey: "url",
    cell: ({ getValue }) => (
      <div className="overflow-hidden">
        <Tooltip title={getValue() as string}>
          <span className="font-mono text-gray-600 text-xs">
            {displayFriendlyUrl(getValue() as string)}
          </span>
        </Tooltip>
      </div>
    ),
  },
  {
    header: "Created",
    accessorKey: "created_at",
    cell: ({ getValue }) => (
      <span>
        {getValue() ? new Date(getValue() as string).toLocaleDateString() : "N/A"}
      </span>
    ),
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
