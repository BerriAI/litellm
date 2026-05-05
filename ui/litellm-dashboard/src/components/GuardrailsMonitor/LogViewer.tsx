import {
  CheckCircleOutlined,
  CloseOutlined,
  DownOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import moment from "moment";
import { Button, Spin } from "antd";
import React, { useState } from "react";
import { uiSpendLogsCall } from "@/components/networking";
import { LogDetailsDrawer } from "@/components/view_logs/LogDetailsDrawer";
import type { LogEntry as ViewLogsLogEntry } from "@/components/view_logs/columns";
import type { LogEntry } from "./mockData";

const actionConfig: Record<
  "blocked" | "passed" | "flagged",
  { icon: React.ElementType; color: string; bg: string; border: string; label: string }
> = {
  blocked: {
    icon: CloseOutlined,
    color: "text-red-600",
    bg: "bg-red-50",
    border: "border-red-200",
    label: "Blocked",
  },
  passed: {
    icon: CheckCircleOutlined,
    color: "text-green-600",
    bg: "bg-green-50",
    border: "border-green-200",
    label: "Passed",
  },
  flagged: {
    icon: WarningOutlined,
    color: "text-amber-600",
    bg: "bg-amber-50",
    border: "border-amber-200",
    label: "Flagged",
  },
};

interface LogViewerProps {
  guardrailName?: string;
  filterAction?: "all" | "blocked" | "passed" | "flagged";
  logs?: LogEntry[];
  logsLoading?: boolean;
  totalLogs?: number;
  accessToken?: string | null;
  startDate?: string;
  endDate?: string;
}

export function LogViewer({
  guardrailName,
  filterAction = "all",
  logs = [],
  logsLoading = false,
  totalLogs,
  accessToken = null,
  startDate = "",
  endDate = "",
}: LogViewerProps) {
  const [sampleSize, setSampleSize] = useState(10);
  const [activeFilter, setActiveFilter] = useState<string>(filterAction);
  const [selectedRequestId, setSelectedRequestId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const filteredLogs = logs.filter(
    (log) => activeFilter === "all" || log.action === activeFilter
  );
  const displayLogs = filteredLogs.slice(0, sampleSize);
  const total = totalLogs ?? logs.length;
  const sampleSizes = [10, 50, 100];
  const filters: Array<"all" | "blocked" | "flagged" | "passed"> = [
    "all",
    "blocked",
    "flagged",
    "passed",
  ];

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

  const handleLogClick = (log: LogEntry) => {
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
              {guardrailName ? `Logs — ${guardrailName}` : "Request Logs"}
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              {logsLoading
                ? "Loading…"
                : logs.length > 0
                  ? `Showing ${displayLogs.length} of ${total} entries`
                  : "No logs for this period. Select a guardrail and date range."}
            </p>
          </div>
          {logs.length > 0 && (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                {filters.map((f) => (
                  <Button
                    key={f}
                    type={activeFilter === f ? "primary" : "default"}
                    size="small"
                    onClick={() => setActiveFilter(f)}
                  >
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </Button>
                ))}
              </div>
              <div className="h-4 w-px bg-gray-200" />
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
          No logs to display. Adjust filters or date range.
        </div>
      )}
      {!logsLoading && displayLogs.length > 0 && (
        <div className="divide-y divide-gray-100">
          {displayLogs.map((log) => {
            const config = actionConfig[log.action];
            const ActionIcon = config.icon;
            return (
              <button
                key={log.id}
                type="button"
                onClick={() => handleLogClick(log)}
                className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors flex items-start gap-3"
              >
                <ActionIcon
                  className={`w-4 h-4 mt-0.5 flex-shrink-0 ${config.color}`}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded border ${config.bg} ${config.color} ${config.border}`}
                    >
                      {config.label}
                    </span>
                    <span className="text-xs text-gray-400">{log.timestamp}</span>
                    <span className="text-xs text-gray-400">·</span>
                    {log.model && (
                      <span className="text-xs text-gray-500">{log.model}</span>
                    )}
                  </div>
                  <p className="text-sm text-gray-800 truncate">
                    {log.input_snippet ?? log.input ?? "—"}
                  </p>
                </div>
                <DownOutlined className="w-4 h-4 text-gray-400 flex-shrink-0 mt-1" />
              </button>
            );
          })}
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
