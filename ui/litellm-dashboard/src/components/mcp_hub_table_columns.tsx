import { ColumnDef } from "@tanstack/react-table";
import { Button, Badge, Text } from "@tremor/react";
import { Tooltip, Tag } from "antd";
import { CopyOutlined, InfoCircleOutlined } from "@ant-design/icons";

export interface MCPServerData {
  server_id: string;
  server_name: string;
  alias?: string | null;
  description?: string | null;
  url: string;
  transport: string;
  auth_type: string;
  credentials?: any;
  created_at: string;
  created_by: string;
  updated_at: string;
  updated_by: string;
  teams: string[];
  mcp_access_groups: string[];
  allowed_tools: string[];
  extra_headers: any[];
  mcp_info: Record<string, any>;
  static_headers: Record<string, any>;
  status: string;
  last_health_check?: string | null;
  health_check_error?: string | null;
  command?: string | null;
  args: string[];
  env: Record<string, any>;
  [key: string]: any;
}

export const mcpHubColumns = (
  showModal: (server: MCPServerData) => void,
  copyToClipboard: (text: string) => void,
  publicPage: boolean = false,
): ColumnDef<MCPServerData>[] => {
  const allColumns: ColumnDef<MCPServerData>[] = [
    {
      header: "Server Name",
      accessorKey: "server_name",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;

        return (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <Text className="font-medium text-sm">{server.server_name}</Text>
              <Tooltip title="Copy server name">
                <CopyOutlined
                  onClick={() => copyToClipboard(server.server_name)}
                  className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
                />
              </Tooltip>
            </div>
            {/* Show description on mobile */}
            <div className="md:hidden">
              <Text className="text-xs text-gray-600">{server.description || "-"}</Text>
            </div>
          </div>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "description",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;

        return (
          <Text className="text-xs line-clamp-2">
            {server.description || "-"}
          </Text>
        );
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: "URL",
      accessorKey: "url",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;

        return (
          <div className="flex items-center space-x-2">
            <Text className="text-xs truncate max-w-xs">{server.url}</Text>
            <Tooltip title="Copy URL">
              <CopyOutlined
                onClick={() => copyToClipboard(server.url)}
                className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs flex-shrink-0"
              />
            </Tooltip>
          </div>
        );
      },
      meta: {
        className: "hidden lg:table-cell",
      },
    },
    {
      header: "Transport",
      accessorKey: "transport",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;

        return (
          <Badge color="blue" size="sm">
            {server.transport}
          </Badge>
        );
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: "Auth Type",
      accessorKey: "auth_type",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;

        const authColor = server.auth_type === "none" ? "gray" : "green";

        return (
          <Badge color={authColor} size="sm">
            {server.auth_type}
          </Badge>
        );
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: "Status",
      accessorKey: "status",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;

        const statusColors: Record<string, string> = {
          active: "green",
          inactive: "red",
          unknown: "gray",
          healthy: "green",
          unhealthy: "red",
        };

        const color = statusColors[server.status] || "gray";

        return (
          <Badge color={color} size="sm">
            {server.status || "unknown"}
          </Badge>
        );
      },
    },
    {
      header: "Tools",
      accessorKey: "allowed_tools",
      enableSorting: false,
      cell: ({ row }) => {
        const server = row.original;
        const tools = server.allowed_tools || [];

        return (
          <div className="space-y-1">
            <Text className="text-xs font-medium">
              {tools.length > 0 ? `${tools.length} tool${tools.length !== 1 ? "s" : ""}` : "All tools"}
            </Text>
            {tools.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {tools.slice(0, 2).map((tool, idx) => (
                  <Tag key={idx} color="purple" className="text-xs">
                    {tool}
                  </Tag>
                ))}
                {tools.length > 2 && (
                  <Text className="text-xs text-gray-500">+{tools.length - 2}</Text>
                )}
              </div>
            )}
          </div>
        );
      },
      meta: {
        className: "hidden lg:table-cell",
      },
    },
    {
      header: "Created By",
      accessorKey: "created_by",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;

        return (
          <Text className="text-xs">
            {server.created_by || "-"}
          </Text>
        );
      },
      meta: {
        className: "hidden xl:table-cell",
      },
    },
    {
      header: "Public",
      accessorKey: "mcp_info.is_public",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const publicA = rowA.original.mcp_info?.is_public === true ? 1 : 0;
        const publicB = rowB.original.mcp_info?.is_public === true ? 1 : 0;
        return publicA - publicB;
      },
      cell: ({ row }) => {
        const server = row.original;

        return server.mcp_info?.is_public === true ? (
          <Badge color="green" size="xs">
            Yes
          </Badge>
        ) : (
          <Badge color="gray" size="xs">
            No
          </Badge>
        );
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: "Details",
      id: "details",
      enableSorting: false,
      cell: ({ row }) => {
        const server = row.original;

        return (
          <Button size="xs" variant="secondary" onClick={() => showModal(server)} icon={InfoCircleOutlined}>
            <span className="hidden lg:inline">Details</span>
            <span className="lg:hidden">Info</span>
          </Button>
        );
      },
    },
  ];

  return allColumns;
};

