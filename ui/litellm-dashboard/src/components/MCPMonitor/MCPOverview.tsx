import { ToolOutlined, TeamOutlined, KeyOutlined, ApiOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Spin } from "antd";
import React from "react";
import { getMCPUsageOverview } from "@/components/networking";

interface MCPOverviewProps {
  accessToken?: string | null;
  startDate: string;
  endDate: string;
  onSelectServer: (serverName: string) => void;
}

export function MCPOverview({
  accessToken,
  startDate,
  endDate,
  onSelectServer,
}: MCPOverviewProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["mcp-usage-overview", startDate, endDate],
    queryFn: () => getMCPUsageOverview(accessToken!, startDate, endDate),
    enabled: !!accessToken,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spin size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12 text-red-600">
        Failed to load MCP usage overview.
      </div>
    );
  }

  const servers = data?.servers ?? [];
  const totalRequests = data?.total_requests ?? 0;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
            <ApiOutlined />
            <span>Total MCP Requests</span>
          </div>
          <div className="text-2xl font-semibold text-gray-900">
            {totalRequests.toLocaleString()}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
            <ToolOutlined />
            <span>Active Servers</span>
          </div>
          <div className="text-2xl font-semibold text-gray-900">
            {servers.length}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
            <TeamOutlined />
            <span>Unique Users</span>
          </div>
          <div className="text-2xl font-semibold text-gray-900">
            {servers.reduce(
              (sum: number, s: any) => sum + (s.unique_users || 0),
              0
            )}
          </div>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-base font-semibold text-gray-900">
            MCP Servers
          </h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Click a server to view detailed logs and tool usage
          </p>
        </div>

        {servers.length === 0 ? (
          <div className="py-12 text-center text-sm text-gray-500">
            No MCP server activity found for this date range.
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {servers.map((server: any) => (
              <button
                key={server.mcp_server_name}
                type="button"
                onClick={() => onSelectServer(server.mcp_server_name)}
                className="w-full text-left px-4 py-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center">
                      <ToolOutlined className="text-blue-600" />
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">
                        {server.mcp_server_name}
                      </div>
                      {server.description && (
                        <div className="text-xs text-gray-500 mt-0.5">
                          {server.description}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-6 text-sm">
                    <div className="text-right">
                      <div className="text-gray-900 font-medium">
                        {server.total_requests.toLocaleString()}
                      </div>
                      <div className="text-xs text-gray-500">requests</div>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center gap-1 text-gray-700">
                        <ToolOutlined className="text-xs" />
                        <span>{server.top_tools?.length ?? 0}</span>
                      </div>
                      <div className="text-xs text-gray-500">tools</div>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center gap-1 text-gray-700">
                        <TeamOutlined className="text-xs" />
                        <span>{server.unique_users}</span>
                      </div>
                      <div className="text-xs text-gray-500">users</div>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center gap-1 text-gray-700">
                        <KeyOutlined className="text-xs" />
                        <span>{server.unique_keys}</span>
                      </div>
                      <div className="text-xs text-gray-500">keys</div>
                    </div>
                  </div>
                </div>

                {server.top_tools && server.top_tools.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {server.top_tools.slice(0, 5).map((tool: any) => (
                      <span
                        key={tool.name}
                        className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded bg-gray-100 text-gray-700"
                      >
                        {tool.name}
                        <span className="ml-1 text-gray-400">
                          ({tool.count})
                        </span>
                      </span>
                    ))}
                    {server.top_tools.length > 5 && (
                      <span className="text-xs text-gray-400">
                        +{server.top_tools.length - 5} more
                      </span>
                    )}
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
