import { ColumnDef } from "@tanstack/react-table";
import { MCPServer } from "./types";
import { Icon } from "@tremor/react";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { getMaskedAndFullUrl } from "./utils";
import { Tooltip } from "antd";
import { CheckOutlined } from "@ant-design/icons";

export const mcpServerColumns = (
  userRole: string,
  onView: (serverId: string) => void,
  onEdit: (serverId: string) => void,
  onDelete: (serverId: string) => void,
  isLoadingHealth?: boolean,
  onByokConnect?: (server: MCPServer) => void,
): ColumnDef<MCPServer>[] => [
  {
    accessorKey: "server_id",
    header: "Server ID",
    enableSorting: true,
    cell: ({ row }) => (
      <button
        onClick={() => onView(row.original.server_id)}
        className="font-mono text-blue-600 bg-blue-50 hover:bg-blue-100 text-xs font-medium px-2 py-0.5 rounded-md border border-blue-200 text-left truncate whitespace-nowrap cursor-pointer max-w-[15ch] transition-colors"
      >
        {row.original.server_id.slice(0, 7)}...
      </button>
    ),
  },
  {
    accessorKey: "server_name",
    header: "Name",
    enableSorting: true,
  },
  {
    accessorKey: "alias",
    header: "Alias",
    enableSorting: true,
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
    enableSorting: true,
    cell: ({ row }) => {
      const transport = row.original.transport || "http";
      const specPath = row.original.spec_path;
      const displayTransport = specPath && transport !== "stdio" ? "OPENAPI" : transport;
      const label = displayTransport.toUpperCase();
      const colorClass =
        label === "HTTP" ? "bg-blue-50 text-blue-700 border-blue-200" :
        label === "SSE" ? "bg-purple-50 text-purple-700 border-purple-200" :
        label === "STDIO" ? "bg-amber-50 text-amber-700 border-amber-200" :
        label === "OPENAPI" ? "bg-teal-50 text-teal-700 border-teal-200" :
        "bg-gray-50 text-gray-700 border-gray-200";
      return (
        <span className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded border ${colorClass}`}>
          {label}
        </span>
      );
    },
  },
  {
    accessorKey: "auth_type",
    header: "Auth Type",
    enableSorting: true,
    cell: ({ getValue }) => {
      const authType = (getValue() as string) || "none";
      const colorClass =
        authType === "oauth2" ? "bg-indigo-50 text-indigo-700 border-indigo-200" :
        authType === "bearer_token" ? "bg-sky-50 text-sky-700 border-sky-200" :
        authType === "api_key" ? "bg-emerald-50 text-emerald-700 border-emerald-200" :
        authType === "basic" ? "bg-orange-50 text-orange-700 border-orange-200" :
        "bg-gray-50 text-gray-500 border-gray-200";
      return (
        <span className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded border ${colorClass}`}>
          {authType}
        </span>
      );
    },
  },
  {
    id: "health_status",
    header: "Health Status",
    cell: ({ row }) => {
      const server = row.original;
      const status = server.status || "unknown";
      const lastCheck = server.last_health_check;
      const error = server.health_check_error;

      if (isLoadingHealth) {
        return (
          <span className="inline-flex items-center gap-1.5 text-xs text-gray-400 px-2 py-0.5 rounded-full bg-gray-50 border border-gray-100">
            <span className="h-1.5 w-1.5 rounded-full bg-gray-300 animate-pulse"></span>
            Checking
          </span>
        );
      }

      const getStatusColor = (status: string) => {
        switch (status) {
          case "healthy":
            return "text-green-700 bg-green-50 border border-green-200";
          case "unhealthy":
            return "text-red-700 bg-red-50 border border-red-200";
          default:
            return "text-gray-600 bg-gray-50 border border-gray-200";
        }
      };

      const getStatusIcon = (status: string) => {
        switch (status) {
          case "healthy":
            return "✓";
          case "unhealthy":
            return "✗";
          default:
            return "?";
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
          <span
            className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full cursor-default ${getStatusColor(status)}`}
          >
            <span>{getStatusIcon(status)}</span>
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </span>
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
        if (typeof groups[0] === "string") {
          const joined = groups.join(", ");
          return (
            <Tooltip title={joined}>
              <div className="flex items-center gap-1 max-w-[200px]">
                <span className="inline-flex items-center text-xs font-medium px-1.5 py-0.5 rounded bg-gray-100 text-gray-700 border border-gray-200 truncate max-w-[140px]">
                  {groups[0]}
                </span>
                {groups.length > 1 && (
                  <span className="text-xs text-gray-400 font-medium">+{groups.length - 1}</span>
                )}
              </div>
            </Tooltip>
          );
        }
      }
      return <span className="text-xs text-gray-400">—</span>;
    },
  },
  {
    id: "available_on_public_internet",
    header: "Network Access",
    cell: ({ row }) => {
      const isPublic = row.original.available_on_public_internet;
      return isPublic ? (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-50 text-green-700 rounded-full border border-green-200 text-xs font-medium">
          <span className="h-1.5 w-1.5 rounded-full bg-green-500"></span>
          Public
        </span>
      ) : (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-orange-50 text-orange-700 rounded-full border border-orange-200 text-xs font-medium">
          <span className="h-1.5 w-1.5 rounded-full bg-orange-500"></span>
          Internal
        </span>
      );
    },
  },
  {
    header: "Created",
    accessorKey: "created_at",
    enableSorting: true,
    sortingFn: "datetime",
    cell: ({ row }) => {
      const server = row.original;
      if (!server.created_at) return <span className="text-xs text-gray-400">—</span>;
      const date = new Date(server.created_at);
      return (
        <Tooltip title={date.toLocaleString()}>
          <span className="text-xs text-gray-600">{date.toLocaleDateString()}</span>
        </Tooltip>
      );
    },
  },
  {
    header: "Updated",
    accessorKey: "updated_at",
    enableSorting: true,
    sortingFn: "datetime",
    cell: ({ row }) => {
      const server = row.original;
      if (!server.updated_at) return <span className="text-xs text-gray-400">—</span>;
      const date = new Date(server.updated_at);
      return (
        <Tooltip title={date.toLocaleString()}>
          <span className="text-xs text-gray-600">{date.toLocaleDateString()}</span>
        </Tooltip>
      );
    },
  },
  {
    id: "byok_credential",
    header: "Credential",
    cell: ({ row }) => {
      const server = row.original;
      if (!server.is_byok) {
        return <span className="text-gray-300 text-xs">—</span>;
      }
      if (server.has_user_credential) {
        return (
          <div className="flex items-center gap-2">
            <span className="text-green-600 text-xs font-medium flex items-center gap-1">
              <CheckOutlined /> Connected
            </span>
            {onByokConnect && (
              <button
                className="text-xs text-gray-400 hover:text-blue-500 underline"
                onClick={() => onByokConnect(server)}
              >
                Reconnect
              </button>
            )}
          </div>
        );
      }
      return onByokConnect ? (
        <button
          className="text-xs bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded-lg font-medium"
          onClick={() => onByokConnect(server)}
        >
          Connect
        </button>
      ) : null;
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
