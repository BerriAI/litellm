import React, { useState } from "react";
import { LogEntry } from "./columns";
import { DataTable } from "./table";
import { columns } from "./columns";
import { Card, Title, Text, Metric, Button as TremorButton } from "@tremor/react";
import { RequestViewer } from "./index";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { Button } from "antd";
import { copyToClipboard as utilCopyToClipboard } from "../../utils/dataUtils";
import { CheckIcon, CopyIcon } from "lucide-react";
import { Tooltip } from "antd";

interface SessionViewProps {
  sessionId: string;
  logs: LogEntry[];
  onBack: () => void;
}

export const SessionView: React.FC<SessionViewProps> = ({ sessionId, logs, onBack }) => {
  // Track which log row is expanded
  const [expandedRequestId, setExpandedRequestId] = useState<string | null>(null);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  // Calculate session metrics
  const totalCost = logs.reduce((sum, log) => sum + (log.spend || 0), 0);
  const totalTokens = logs.reduce((sum, log) => sum + (log.total_tokens || 0), 0);

  // Calculate cache token totals from metadata
  const totalCacheReadTokens = logs.reduce((sum, log) => {
    const cacheReadTokens = log.metadata?.additional_usage_values?.cache_read_input_tokens || 0;
    return sum + cacheReadTokens;
  }, 0);

  const totalCacheCreationTokens = logs.reduce((sum, log) => {
    const cacheCreationTokens = log.metadata?.additional_usage_values?.cache_creation_input_tokens || 0;
    return sum + cacheCreationTokens;
  }, 0);

  // Calculate total tokens including cache tokens
  const totalTokensWithCache = totalTokens + totalCacheReadTokens + totalCacheCreationTokens;

  const startTime = logs.length > 0 ? new Date(logs[0].startTime) : new Date();
  const endTime = logs.length > 0 ? new Date(logs[logs.length - 1].endTime) : new Date();
  const durationMs = endTime.getTime() - startTime.getTime();
  const durationSec = (durationMs / 1000).toFixed(2);

  // Prepare data for the timeline chart
  const timelineData = logs.map((log) => ({
    time: new Date(log.startTime).toISOString(),
    tokens: log.total_tokens || 0,
    cost: log.spend || 0,
  }));

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header with back button */}
      <div className="mb-8">
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onBack} className="mb-4">
          Back to All Logs
        </TremorButton>
        <div className="mt-4">
          <h1 className="text-2xl font-semibold text-gray-900">Session Details</h1>
          <div className="space-y-2">
            <div className="flex items-center cursor-pointer">
              <p className="text-sm text-gray-500 font-mono">{sessionId}</p>
              <Button
                type="text"
                size="small"
                icon={copiedStates["session-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
                onClick={() => copyToClipboard(sessionId, "session-id")}
                className={`left-2 z-10 transition-all duration-200 ${
                  copiedStates["session-id"]
                    ? "text-green-600 bg-green-50 border-green-200"
                    : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                }`}
              />
            </div>
            <a
              href="https://docs.litellm.ai/docs/proxy/ui_logs_sessions"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              Get started with session management here
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
            </a>
          </div>
        </div>
      </div>

      {/* Session Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <Text>Total Requests</Text>
          <Metric>{logs.length}</Metric>
        </Card>
        <Card>
          <Text>Total Cost</Text>
          <Metric>${formatNumberWithCommas(totalCost, 6)}</Metric>
        </Card>
        <Tooltip
          title={
            <div className="text-white min-w-[200px]">
              <div className="text-lg font-medium mb-3">Usage breakdown</div>
              <div className="space-y-4">
                <div>
                  <div className="text-base font-medium mb-2">Input usage:</div>
                  <div className="space-y-2 text-sm text-gray-300">
                    <div className="flex justify-between">
                      <span>input:</span>
                      <span className="ml-8">
                        {formatNumberWithCommas(logs.reduce((sum, log) => sum + (log.prompt_tokens || 0), 0))}
                      </span>
                    </div>
                    {totalCacheReadTokens > 0 && (
                      <div className="flex justify-between">
                        <span>input_cached_tokens:</span>
                        <span className="ml-8">{formatNumberWithCommas(totalCacheReadTokens)}</span>
                      </div>
                    )}
                    {totalCacheCreationTokens > 0 && (
                      <div className="flex justify-between">
                        <span>input_cache_creation_tokens:</span>
                        <span className="ml-8">{formatNumberWithCommas(totalCacheCreationTokens)}</span>
                      </div>
                    )}
                  </div>
                </div>
                <div className="border-t border-gray-600 pt-3">
                  <div className="text-base font-medium mb-2">Output usage:</div>
                  <div className="space-y-2 text-sm text-gray-300">
                    <div className="flex justify-between">
                      <span>output:</span>
                      <span className="ml-8">
                        {formatNumberWithCommas(logs.reduce((sum, log) => sum + (log.completion_tokens || 0), 0))}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="border-t border-gray-600 pt-3">
                  <div className="flex justify-between items-center">
                    <span className="text-base font-medium">Total usage:</span>
                    <span className="text-sm text-gray-300">{formatNumberWithCommas(totalTokensWithCache)}</span>
                  </div>
                </div>
              </div>
            </div>
          }
          placement="top"
          overlayStyle={{ minWidth: "300px" }}
        >
          <Card>
            <div className="flex items-center justify-between">
              <Text>Total Tokens</Text>
              <span className="text-gray-400 text-sm">ⓘ</span>
            </div>
            <Metric>{formatNumberWithCommas(totalTokensWithCache)}</Metric>
          </Card>
        </Tooltip>
      </div>
      {/* Request Timeline */}
      <Title>Session Logs</Title>
      <div className="mt-4">
        <DataTable
          columns={columns}
          data={logs}
          renderSubComponent={RequestViewer}
          getRowCanExpand={() => true}
          loadingMessage="Loading logs..."
          noDataMessage="No logs found"
        />
      </div>
    </div>
  );
};
