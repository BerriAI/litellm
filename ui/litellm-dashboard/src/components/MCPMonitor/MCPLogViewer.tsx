import {
  ToolOutlined,
  DownOutlined,
  UserOutlined,
  KeyOutlined,
} from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import moment from "moment";
import { Button, Spin } from "antd";
import React, { useState } from "react";
import { uiSpendLogsCall } from "@/components/networking";
import { LogDetailsDrawer } from "@/components/view_logs/LogDetailsDrawer";
import type { LogEntry as ViewLogsLogEntry } from "@/components/view_logs/columns";

interface MCPLogEntry {
  id: string;
  timestamp: string;
  mcp_server_name: string;
  tool_name?: string;
  api_key_hash?: string;
  api_key_alias?: string;
  user_id?: string;
  team_id?: string;
  model?: string;
  status?: string;
  spend?: number;
  input_snippet?: string;
  output_snippet?: string;
}

interface MCPLogViewerProps {
  serverName?: string;
  logs: MCPLogEntry[];
  logsLoading?: boolean;
  totalLogs?: number;
  accessToken?: string | null;
  startDate?: string;
  endDate?: string;
}

export function MCPLogViewer({
  serverName,
  logs = [],
  logsLoading = false,
  totalLogs,
  accessToken = null,
  startDate = "",
  endDate = "",
}: MCPLogViewerProps) {
  const [sampleSize, setSampleSize] = useState(10);
  const [selectedRequestId, setSelectedRequestId] = useState<string | null>(
    null
  );
  const [drawerOpen, setDrawerOpen] = useState(false);

  const displayLogs = logs.slice(0, sampleSize);
  const total = totalLogs ?? logs.length;
  const sampleSizes = [10, 50, 100];

  const startTime = startDate
    ? moment(startDate).utc().format("YYYY-MM-DD HH:mm:ss")
    : moment().subtract(24, "hours").utc().format("YYYY-MM-DD HH:mm:ss");
  const endTime = endDate
    ? moment(endDate).utc().endOf("day").format("YYYY-MM-DD HH:mm:ss")
    : moment().utc().format("YYYY-MM-DD HH:mm:ss");

  const { data: fullLogResponse } = useQuery({
    queryKey: ["spend-log-by-request", selectedRequestId, startTime, endTime],
    queryFn: async () => {
      if (!accessToken || !selectedRequestId) return null;
      const res = await uiSpendLogsCall({
        accessToken,
        start_date: startTime,
        end_date: endTime,
        page: 1,
        page_size: 10,
        params: { request_id: selectedRequestId },
      });
      return res as { data: ViewLogsLogEntry[]; total: number };
    },
    enabled: Boolean(accessToken && selectedRequestId && drawerOpen),
  });

  const selectedLog: ViewLogsLogEntry | null =
    fullLogResponse?.data?.[0] ?? null;

  const handleLogClick = (log: MCPLogEntry) => {
    setSelectedRequestId(log.id);
    setDrawerOpen(true);
  };

  const handleCloseDrawer = () => {
    setDrawerOpen(false);
    setSelectedRequestId(null);
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-base font-semibold text-gray-900">
              {serverName ? `Logs — ${serverName}` : "MCP Request Logs"}
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              {logsLoading
                ? "Loading…"
                : logs.length > 0
                  ? `Showing ${displayLogs.length} of ${total} entries`
                  : "No logs for this period."}
            </p>
          </div>
          {logs.length > 0 && (
            <div className="flex items-center gap-1">
              <span className="text-xs text-gray-500 mr-1">Sample:</span>
              {sampleSizes.map((size) => (
                <Button
                  key={size}
                  type={sampleSize === size ? "primary" : "default"}
                  size="small"
                  onClick={() => setSampleSize(size)}
                >
                  {size}
                </Button>
              ))}
            </div>
          )}
        </div>
      </div>

      {logsLoading && (
        <div className="flex items-center justify-center py-12">
          <Spin />
        </div>
      )}
      {!logsLoading && displayLogs.length === 0 && (
        <div className="py-12 text-center text-sm text-gray-500">
          No logs to display. Adjust date range.
        </div>
      )}
      {!logsLoading && displayLogs.length > 0 && (
        <div className="divide-y divide-gray-100">
          {displayLogs.map((log) => (
            <button
              key={log.id}
              type="button"
              onClick={() => handleLogClick(log)}
              className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors flex items-start gap-3"
            >
              <ToolOutlined className="w-4 h-4 mt-0.5 flex-shrink-0 text-blue-500" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded bg-blue-50 text-blue-700 border border-blue-200">
                    {log.tool_name || "unknown"}
                  </span>
                  <span className="text-xs text-gray-400">
                    {log.timestamp}
                  </span>
                  {log.status && (
                    <>
                      <span className="text-xs text-gray-400">·</span>
                      <span
                        className={`text-xs ${
                          log.status === "success"
                            ? "text-green-600"
                            : "text-red-600"
                        }`}
                      >
                        {log.status}
                      </span>
                    </>
                  )}
                  {log.user_id && (
                    <>
                      <span className="text-xs text-gray-400">·</span>
                      <span className="inline-flex items-center gap-0.5 text-xs text-gray-500">
                        <UserOutlined className="text-[10px]" />
                        {log.user_id}
                      </span>
                    </>
                  )}
                  {(log.api_key_alias || log.api_key_hash) && (
                    <>
                      <span className="text-xs text-gray-400">·</span>
                      <span className="inline-flex items-center gap-0.5 text-xs text-gray-500">
                        <KeyOutlined className="text-[10px]" />
                        {log.api_key_alias ||
                          (log.api_key_hash
                            ? `${log.api_key_hash.slice(0, 8)}...`
                            : "")}
                      </span>
                    </>
                  )}
                </div>
                <p className="text-sm text-gray-800 truncate">
                  {log.input_snippet ?? "—"}
                </p>
              </div>
              <DownOutlined className="w-4 h-4 text-gray-400 flex-shrink-0 mt-1" />
            </button>
          ))}
        </div>
      )}

      <LogDetailsDrawer
        open={drawerOpen}
        onClose={handleCloseDrawer}
        logEntry={selectedLog}
        accessToken={accessToken}
        allLogs={selectedLog ? [selectedLog] : []}
        startTime={startTime}
      />
    </div>
  );
}
