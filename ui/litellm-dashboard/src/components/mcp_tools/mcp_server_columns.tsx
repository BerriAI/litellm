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
  onDelete: (serverId: string) => void,
  isLoadingHealth?: boolean,
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
    accessorKey: "server_name",
    header: "Name",
  },
  {
    accessorKey: "alias",
    header: "Alias",
  },
  {
    id: "url",
    header: "URL",
    cell: ({ row }) => {
      const url = row.original.url;
      if (!url) {
        return <span className="text-gray-400">—</span>;
      }
      const { maskedUrl } = getMaskedAndFullUrl(url);
      return <span className="font-mono text-sm">{maskedUrl}</span>;
    },
  },
  {
    accessorKey: "transport",
    header: "Transport",
    cell: ({ getValue }) => <span>{((getValue() as string) || "http").toUpperCase()}</span>,
  },
  {
    accessorKey: "auth_type",
    header: "Auth Type",
    cell: ({ getValue }) => <span>{(getValue() as string) || "none"}</span>,
  },
  {
    id: "health_status",
    header: "Health Status",
    cell: ({ row }) => {
      const server = row.original;
      const status = server.status || "unknown";
      const lastCheck = server.last_health_check;
      const error = server.health_check_error;

      // Show loading spinner if health check is in progress
      if (isLoadingHealth) {
        return (
          <div className="flex items-center text-gray-500">
            <svg className="animate-spin h-4 w-4 mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span className="text-xs">Loading...</span>
          </div>
        );
      }

      const getStatusColor = (status: string) => {
        switch (status) {
          case "healthy":
            return "text-green-500 bg-green-50 hover:bg-green-100";
          case "unhealthy":
            return "text-red-500 bg-red-50 hover:bg-red-100";
          default:
            return "text-gray-500 bg-gray-50 hover:bg-gray-100";
        }
      };

      const getStatusIcon = (status: string) => {
        switch (status) {
          case "healthy":
            return "●";
          case "unhealthy":
            return "●";
          default:
            return "●";
        }
      };

      const tooltipContent = (
        <div className="max-w-xs">
          <div className="font-semibold mb-1">Health Status: {status}</div>
          {lastCheck && <div className="text-xs mb-1">Last Check: {new Date(lastCheck).toLocaleString()}</div>}
          {error && (
            <div className="text-xs">
              <div className="font-medium text-red-400 mb-1">Error:</div>
              <div className="break-words">{error}</div>
            </div>
          )}
          {!lastCheck && !error && <div className="text-xs text-gray-400">No health check data available</div>}
        </div>
      );

      return (
        <Tooltip title={tooltipContent} placement="top">
          <button
            className={`font-mono text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[10ch] ${getStatusColor(status)}`}
          >
            <span className="mr-1">{getStatusIcon(status)}</span>
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        </Tooltip>
      );
    },
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
              <span className="max-w-[200px] truncate block">
                {joined.length > 30 ? `${joined.slice(0, 30)}...` : joined}
              </span>
            </Tooltip>
          );
        }
      }
      return <span className="text-gray-400 italic">None</span>;
    },
  },
  {
    id: "available_on_public_internet",
    header: "Network Access",
    cell: ({ row }) => {
      const isPublic = row.original.available_on_public_internet;
      return isPublic ? (
        <span className="px-2 py-0.5 bg-green-50 text-green-700 rounded text-xs font-medium">Public</span>
      ) : (
        <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs font-medium">Internal</span>
      );
    },
  },
  {
    header: "Created At",
    accessorKey: "created_at",
    sortingFn: "datetime",
    cell: ({ row }) => {
      const server = row.original;
      return (
        <span className="text-xs">{server.created_at ? new Date(server.created_at).toLocaleDateString() : "-"}</span>
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
        <span className="text-xs">{server.updated_at ? new Date(server.updated_at).toLocaleDateString() : "-"}</span>
      );
    },
  },
  {
    id: "actions",
    header: "Actions",
    cell: ({ row }) => (
      <div className="flex items-center gap-2">
        <Tooltip title="Edit MCP Server">
          <Icon
            icon={PencilAltIcon}
            size="sm"
            onClick={() => onEdit(row.original.server_id)}
            className="cursor-pointer hover:text-blue-600"
          />
        </Tooltip>
        <Tooltip title="Delete MCP Server">
          <Icon
            icon={TrashIcon}
            size="sm"
            onClick={() => onDelete(row.original.server_id)}
            className="cursor-pointer hover:text-red-600"
          />
        </Tooltip>
      </div>
    ),
  },
];
