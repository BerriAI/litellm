import { useQuery } from "@tanstack/react-query";
import { Card, LineChart, Title, Text } from "@tremor/react";
import { Spin } from "antd";
import {
  ArrowLeft,
  ShieldCheck,
} from "lucide-react";
import moment from "moment";
import React, { useState } from "react";
import { uiSpendLogsCall } from "@/components/networking";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { LogDetailsDrawer } from "@/components/view_logs/LogDetailsDrawer";
import type { LogEntry as SpendLogEntry } from "@/components/view_logs/columns";
import type { AgentData } from "./AgentTable";

interface AgentDetailProps {
  agent: AgentData;
  onClose: () => void;
  accessToken?: string | null;
}

function generateDriftData(baseScore: number, trend: "up" | "down" | "flat") {
  const data: Array<{ date: string; purpose: number; tone: number; hallucination: number }> = [];
  let currentPurpose = baseScore;
  let currentTone = baseScore * 0.8;
  let currentHallucination = baseScore * 0.5;

  for (let i = 14; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    data.push({
      date: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      purpose: Math.max(0, currentPurpose + (Math.random() * 0.05 - 0.025)),
      tone: Math.max(0, currentTone + (Math.random() * 0.04 - 0.02)),
      hallucination: Math.max(0, currentHallucination + (Math.random() * 0.03 - 0.015)),
    });
    if (trend === "up") {
      currentPurpose += 0.02;
      currentTone += 0.015;
      currentHallucination += 0.01;
    } else if (trend === "down") {
      currentPurpose -= 0.01;
      currentTone -= 0.01;
      currentHallucination -= 0.005;
    }
  }
  return data;
}

function getAgentDetails(agent: AgentData) {
  const isCritical = agent.status === "Critical" || agent.status === "Killed";
  const isWarning = agent.status === "Warning";
  const score = Math.round(agent.driftScore * 100);

  const trend: "up" | "down" | "flat" = isCritical ? "up" : isWarning ? "up" : "flat";

  return {
    score,
    driftData: generateDriftData(agent.driftScore, trend),
    factors: {
      purpose: isCritical ? 45 : isWarning ? 72 : 98,
      stability: isCritical ? 30 : isWarning ? 65 : 95,
      compliance: isCritical ? 20 : isWarning ? 85 : 100,
      resistance: isCritical ? 60 : isWarning ? 90 : 99,
    },
  };
}

function FactorBar({ label, value }: { label: string; value: number }) {
  const color = value < 50 ? "bg-red-500" : value < 80 ? "bg-amber-500" : "bg-emerald-500";
  const textColor = value < 50 ? "text-red-500" : value < 80 ? "text-amber-500" : "text-emerald-500";

  return (
    <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{label}</span>
        <span className={`text-sm font-bold ${textColor}`}>{value}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function SpendLogRow({ log, onClick }: { log: SpendLogEntry; onClick: () => void }) {
  const statusDot =
    log.status === "failure"
      ? "bg-red-500"
      : log.status === "success" || !log.status
        ? "bg-emerald-500"
        : "bg-gray-400";

  const providerInfo = log.custom_llm_provider
    ? getProviderLogoAndName(log.custom_llm_provider)
    : null;

  const durationMs = log.request_duration_ms ?? (
    log.startTime && log.endTime
      ? Date.parse(log.endTime) - Date.parse(log.startTime)
      : null
  );

  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors flex items-center gap-3"
    >
      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDot}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5 flex-wrap">
          <span className="text-xs text-gray-400">
            {moment(log.startTime).format("MMM D, HH:mm:ss")}
          </span>
          {providerInfo && (
            <>
              <span className="text-xs text-gray-300">&middot;</span>
              <span className="text-xs text-gray-500">{providerInfo.name}</span>
            </>
          )}
          <span className="text-xs text-gray-300">&middot;</span>
          <span className="text-xs font-medium text-gray-700">{log.model}</span>
          {log.call_type && (
            <>
              <span className="text-xs text-gray-300">&middot;</span>
              <span className="inline-flex items-center px-1.5 py-0.5 text-[10px] font-medium rounded bg-gray-100 text-gray-600 border border-gray-200">
                {log.call_type}
              </span>
            </>
          )}
        </div>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          {log.spend != null && (
            <span>${log.spend.toFixed(6)}</span>
          )}
          {log.total_tokens != null && log.total_tokens > 0 && (
            <span>{log.total_tokens} tokens</span>
          )}
          {durationMs != null && (
            <span>{(durationMs / 1000).toFixed(2)}s</span>
          )}
          {log.status === "failure" && (
            <span className="text-red-500 font-medium">Failed</span>
          )}
        </div>
      </div>
      <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </button>
  );
}

export const AgentDetail: React.FC<AgentDetailProps> = ({ agent, onClose, accessToken = null }) => {
  const details = getAgentDetails(agent);
  const [activeTab, setActiveTab] = useState<"overview" | "logs">("overview");
  const [selectedLog, setSelectedLog] = useState<SpendLogEntry | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 50;

  const startTime = moment().subtract(7, "days").utc().format("YYYY-MM-DD HH:mm:ss");
  const endTime = moment().utc().format("YYYY-MM-DD HH:mm:ss");

  const { data: logsResponse, isLoading: logsLoading } = useQuery({
    queryKey: ["agent-spend-logs", agent.id, currentPage, pageSize, startTime, endTime],
    queryFn: async () => {
      if (!accessToken) return { data: [], total: 0, page: 1, page_size: pageSize, total_pages: 0 };
      return await uiSpendLogsCall({
        accessToken,
        start_date: startTime,
        end_date: endTime,
        page: currentPage,
        page_size: pageSize,
        params: {
          agent_id: agent.id,
          sort_by: "startTime",
          sort_order: "desc",
        },
      });
    },
    enabled: !!accessToken && activeTab === "logs",
  });

  const logs: SpendLogEntry[] = logsResponse?.data ?? [];
  const totalLogs = logsResponse?.total ?? 0;
  const totalPages = logsResponse?.total_pages ?? 0;

  const handleLogClick = (log: SpendLogEntry) => {
    setSelectedLog(log);
    setDrawerOpen(true);
  };

  const handleCloseDrawer = () => {
    setDrawerOpen(false);
    setSelectedLog(null);
  };

  const avgLatency = agent.status === "Critical" ? "450ms" : agent.status === "Warning" ? "280ms" : "120ms";

  const statusStyle = agent.status === "Healthy"
    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : agent.status === "Warning"
      ? "bg-amber-50 text-amber-700 border-amber-200"
      : agent.status === "Critical"
        ? "bg-red-50 text-red-700 border-red-200"
        : "bg-gray-50 text-gray-700 border-gray-200";

  const dotStyle = agent.status === "Healthy"
    ? "bg-emerald-500"
    : agent.status === "Warning"
      ? "bg-amber-500"
      : agent.status === "Critical"
        ? "bg-red-500 animate-pulse"
        : "bg-gray-400";

  return (
    <div className="pb-20">
      {/* Header */}
      <div className="max-w-[1200px] mx-auto pt-8 pb-6">
        <button
          onClick={onClose}
          className="flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 transition-colors mb-6"
        >
          <ArrowLeft className="h-4 w-4 mr-1.5" />
          Back to Overview
        </button>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center space-x-3 mb-2">
              <ShieldCheck className="h-6 w-6 text-gray-400" />
              <h1 className="text-2xl font-bold text-gray-900">{agent.name}</h1>
              <div className={`flex items-center px-2.5 py-1 rounded-full text-xs font-medium border ${statusStyle}`}>
                <div className={`w-1.5 h-1.5 rounded-full mr-1.5 ${dotStyle}`} />
                {agent.status}
              </div>
            </div>
            <p className="text-sm text-gray-500 ml-9">
              {agent.type} &bull; Evaluates prompts and responses for behavioral drift and policy violations
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-3 py-1 rounded-md text-xs font-mono font-medium bg-blue-50 text-blue-700 border border-blue-200">
              {agent.type.toLowerCase().replace(" ", "_")}
            </span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="max-w-[1200px] mx-auto border-b border-gray-200">
        <div className="flex space-x-8">
          {(["overview", "logs"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`pb-4 text-sm font-medium transition-colors relative capitalize ${activeTab === tab ? "text-gray-900" : "text-gray-500 hover:text-gray-700"}`}
            >
              {tab}
              {activeTab === tab && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600" />}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="max-w-[1200px] mx-auto pt-8">
        {activeTab === "overview" ? (
          <div className="space-y-10">
            {/* Stat Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="bg-white border border-gray-200">
                <Text className="text-sm font-medium text-gray-600 mb-2">Drift Score</Text>
                <Title className={`text-4xl font-bold ${details.score > 40 ? "text-red-600" : details.score > 20 ? "text-amber-500" : "text-gray-900"}`}>
                  {details.score}
                </Title>
                <Text className="text-sm text-gray-500 mt-1">out of 100</Text>
              </Card>
              <Card className="bg-white border border-gray-200">
                <Text className="text-sm font-medium text-gray-600 mb-2">Total Requests</Text>
                <Title className="text-4xl font-bold text-gray-900">
                  {totalLogs.toLocaleString()}
                </Title>
                <Text className="text-sm text-gray-500 mt-1">last 7 days</Text>
              </Card>
              <Card className="bg-white border border-gray-200">
                <Text className="text-sm font-medium text-gray-600 mb-2">Avg. Latency Added</Text>
                <Title className="text-4xl font-bold text-gray-900">{avgLatency}</Title>
                <Text className="text-sm text-gray-500 mt-1">last 24h</Text>
              </Card>
            </div>

            {/* Drift Chart */}
            <div>
              <Title className="text-lg font-semibold text-gray-900 mb-6">Behavioral Drift</Title>
              <div className="h-[300px] mb-8">
                <LineChart
                  data={details.driftData}
                  index="date"
                  categories={["purpose", "tone", "hallucination"]}
                  colors={["blue", "amber", "red"]}
                  valueFormatter={(v) => v.toFixed(2)}
                  yAxisWidth={48}
                  showLegend={true}
                  curveType="natural"
                  connectNulls={true}
                />
              </div>

              {/* Factor Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <FactorBar label="Purpose Alignment" value={details.factors.purpose} />
                <FactorBar label="Behavioral Stability" value={details.factors.stability} />
                <FactorBar label="Data Compliance" value={details.factors.compliance} />
                <FactorBar label="Attack Resistance" value={details.factors.resistance} />
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white border border-gray-200 rounded-lg">
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div>
                  <h3 className="text-base font-semibold text-gray-900">
                    Logs &mdash; {agent.name}
                  </h3>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {logsLoading
                      ? "Loading\u2026"
                      : logs.length > 0
                        ? `Showing ${logs.length} of ${totalLogs} entries`
                        : "No logs for this period."}
                  </p>
                </div>
                {totalPages > 1 && (
                  <div className="flex items-center gap-2">
                    <button
                      className="px-3 py-1 text-xs font-medium rounded border border-gray-200 text-gray-600 hover:bg-gray-50 disabled:opacity-40"
                      disabled={currentPage <= 1}
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    >
                      Prev
                    </button>
                    <span className="text-xs text-gray-500">
                      Page {currentPage} of {totalPages}
                    </span>
                    <button
                      className="px-3 py-1 text-xs font-medium rounded border border-gray-200 text-gray-600 hover:bg-gray-50 disabled:opacity-40"
                      disabled={currentPage >= totalPages}
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>
            </div>

            {logsLoading && (
              <div className="flex items-center justify-center py-12">
                <Spin />
              </div>
            )}
            {!logsLoading && logs.length === 0 && (
              <div className="py-12 text-center text-sm text-gray-500">
                No logs to display. Adjust filters or date range.
              </div>
            )}
            {!logsLoading && logs.length > 0 && (
              <div className="divide-y divide-gray-100">
                {logs.map((log) => (
                  <SpendLogRow
                    key={log.request_id}
                    log={log}
                    onClick={() => handleLogClick(log)}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <LogDetailsDrawer
        open={drawerOpen}
        onClose={handleCloseDrawer}
        logEntry={selectedLog}
        accessToken={accessToken}
        allLogs={logs}
        onSelectLog={setSelectedLog}
        startTime={startTime}
      />
    </div>
  );
};
