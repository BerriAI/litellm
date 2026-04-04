import {
  ArrowLeftOutlined,
  ToolOutlined,
  TeamOutlined,
  KeyOutlined,
} from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Button, Spin, Tabs } from "antd";
import React, { useMemo, useState } from "react";
import {
  getMCPUsageLogs,
  getMCPUsageTools,
} from "@/components/networking";
import { MCPLogViewer } from "./MCPLogViewer";
import { MCPToolUsersTable } from "./MCPToolUsersTable";

interface MCPServerDetailProps {
  serverName: string;
  onBack: () => void;
  accessToken?: string | null;
  startDate: string;
  endDate: string;
}

export function MCPServerDetail({
  serverName,
  onBack,
  accessToken = null,
  startDate,
  endDate,
}: MCPServerDetailProps) {
  const [activeTab, setActiveTab] = useState("logs");
  const [logsPage, setLogsPage] = useState(1);
  const logsPageSize = 50;

  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: [
      "mcp-usage-logs",
      serverName,
      logsPage,
      logsPageSize,
      startDate,
      endDate,
    ],
    queryFn: () =>
      getMCPUsageLogs(accessToken!, {
        mcpServerName: serverName,
        page: logsPage,
        pageSize: logsPageSize,
        startDate,
        endDate,
      }),
    enabled: !!accessToken && !!serverName,
  });

  const { data: toolsData, isLoading: toolsLoading } = useQuery({
    queryKey: ["mcp-usage-tools", serverName, startDate, endDate],
    queryFn: () =>
      getMCPUsageTools(accessToken!, serverName, startDate, endDate),
    enabled: !!accessToken && !!serverName,
  });

  const logs = useMemo(() => logsData?.logs ?? [], [logsData]);
  const toolEntries = useMemo(
    () => toolsData?.entries ?? [],
    [toolsData]
  );

  const totalLogs = logsData?.total ?? 0;
  const totalToolEntries = toolsData?.total ?? 0;

  const toolsSummary = useMemo(() => {
    const toolMap = new Map<string, number>();
    for (const entry of toolEntries) {
      const current = toolMap.get(entry.tool_name) ?? 0;
      toolMap.set(entry.tool_name, current + entry.call_count);
    }
    return Array.from(toolMap.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([name, count]) => ({ name, count }));
  }, [toolEntries]);

  const uniqueUsers = useMemo(() => {
    const users = new Set<string>();
    for (const entry of toolEntries) {
      if (entry.user_id) users.add(entry.user_id);
    }
    return users.size;
  }, [toolEntries]);

  const uniqueKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const entry of toolEntries) {
      if (entry.api_key_hash) keys.add(entry.api_key_hash);
    }
    return keys.size;
  }, [toolEntries]);

  return (
    <div>
      <div className="mb-6">
        <Button
          type="link"
          icon={<ArrowLeftOutlined />}
          onClick={onBack}
          className="pl-0 mb-4"
        >
          Back to Overview
        </Button>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <ToolOutlined className="text-xl text-blue-500" />
              <h1 className="text-xl font-semibold text-gray-900">
                {serverName}
              </h1>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mt-4">
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <div className="text-xs text-gray-500">Total Requests</div>
            <div className="text-lg font-semibold text-gray-900">
              {totalLogs.toLocaleString()}
            </div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <div className="text-xs text-gray-500">
              <ToolOutlined className="mr-1" />
              Tools Used
            </div>
            <div className="text-lg font-semibold text-gray-900">
              {toolsSummary.length}
            </div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <div className="text-xs text-gray-500">
              <TeamOutlined className="mr-1" />
              Unique Users
            </div>
            <div className="text-lg font-semibold text-gray-900">
              {uniqueUsers}
            </div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <div className="text-xs text-gray-500">
              <KeyOutlined className="mr-1" />
              Unique Keys
            </div>
            <div className="text-lg font-semibold text-gray-900">
              {uniqueKeys}
            </div>
          </div>
        </div>

        {toolsSummary.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {toolsSummary.slice(0, 8).map((tool) => (
              <span
                key={tool.name}
                className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded bg-blue-50 text-blue-700 border border-blue-200"
              >
                {tool.name}
                <span className="ml-1 text-blue-400">
                  ({tool.count})
                </span>
              </span>
            ))}
          </div>
        )}
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          { key: "logs", label: "Request Logs" },
          {
            key: "users",
            label: "Users & Keys",
          },
        ]}
      />

      {activeTab === "logs" && (
        <div className="mt-4">
          <MCPLogViewer
            serverName={serverName}
            logs={logs}
            logsLoading={logsLoading}
            totalLogs={totalLogs}
            accessToken={accessToken}
            startDate={startDate}
            endDate={endDate}
          />
        </div>
      )}

      {activeTab === "users" && (
        <div className="mt-4">
          {toolsLoading ? (
            <div className="flex items-center justify-center py-12">
              <Spin />
            </div>
          ) : (
            <MCPToolUsersTable
              entries={toolEntries}
              total={totalToolEntries}
            />
          )}
        </div>
      )}
    </div>
  );
}
