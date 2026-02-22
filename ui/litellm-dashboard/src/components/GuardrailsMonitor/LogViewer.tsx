import {
  CheckCircleOutlined,
  CloseOutlined,
  CopyOutlined,
  DownOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { Button, Spin } from "antd";
import React, { useState } from "react";
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
}

export function LogViewer({
  guardrailName,
  filterAction = "all",
  logs = [],
  logsLoading = false,
  totalLogs,
}: LogViewerProps) {
  const [sampleSize, setSampleSize] = useState(10);
  const [expandedLog, setExpandedLog] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<string>(filterAction);

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
          const isExpanded = expandedLog === log.id;
          return (
            <div key={log.id}>
              <button
                type="button"
                onClick={() => setExpandedLog(isExpanded ? null : log.id)}
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
                <span
                  className={`flex-shrink-0 mt-1 transition-transform ${
                    isExpanded ? "rotate-180" : ""
                  }`}
                >
                  <DownOutlined className="w-4 h-4 text-gray-400" />
                </span>
              </button>

              {isExpanded && (
                <div className="px-4 pb-4 pl-11">
                  <div className="bg-gray-50 rounded-lg p-4 space-y-3 text-sm">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                          Input
                        </span>
                        <Button
                          type="text"
                          size="small"
                          icon={<CopyOutlined />}
                          aria-label="Copy input"
                        />
                      </div>
                      <p className="text-gray-800 font-mono text-xs bg-white rounded border border-gray-200 p-3">
                        {log.input_snippet ?? log.input ?? "—"}
                      </p>
                    </div>
                    <div>
                      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                        Output
                      </span>
                      <p className="text-gray-800 font-mono text-xs bg-white rounded border border-gray-200 p-3 mt-1">
                        {log.output_snippet ?? log.output ?? "—"}
                      </p>
                    </div>
                    {(log.reason ?? log.score != null) && (
                    <div>
                      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                        Reason
                      </span>
                      <p className="text-gray-700 text-xs mt-1">
                        {log.reason ?? (log.score != null ? `Score: ${log.score}` : "—")}
                      </p>
                    </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      )}
    </div>
  );
}
