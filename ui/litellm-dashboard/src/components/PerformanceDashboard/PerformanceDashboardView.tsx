"use client";

import React, { useRef, useState, useEffect } from "react";
import { Collapse, Spin, Tag } from "antd";
import { DashboardOutlined, CheckCircleOutlined } from "@ant-design/icons";
import { BarChart, LineChart, Card } from "@tremor/react";
import { MetricCard } from "@/components/GuardrailsMonitor/MetricCard";
import {
  usePerformanceSummary,
  PerformanceSummaryResponse,
  PerformanceIssue,
} from "@/app/(dashboard)/hooks/performanceSummary/usePerformanceSummary";

const MAX_HISTORY = 60; // 60 × 10s = 10 min

interface HistoryPoint {
  time: string;
  "Overhead avg"?: number;
  "In-flight"?: number;
  "HTTP pool %"?: number;
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

const SEVERITY_DOT: Record<string, string> = {
  critical: "bg-red-500",
  warning: "bg-amber-400",
  info: "bg-blue-400",
};

export default function PerformanceDashboardView() {
  const { data, isLoading, error, dataUpdatedAt } = usePerformanceSummary();

  // Client-side ring buffer for time-series charts
  const overheadHistoryRef = useRef<HistoryPoint[]>([]);
  const [overheadHistory, setOverheadHistory] = useState<HistoryPoint[]>([]);

  const inflightHistoryRef = useRef<HistoryPoint[]>([]);
  const [inflightHistory, setInflightHistory] = useState<HistoryPoint[]>([]);

  const httpHistoryRef = useRef<HistoryPoint[]>([]);
  const [httpHistory, setHttpHistory] = useState<HistoryPoint[]>([]);

  // Live "X seconds ago" counter
  const [secondsAgo, setSecondsAgo] = useState(0);
  useEffect(() => {
    if (!dataUpdatedAt) return;
    setSecondsAgo(0);
    const interval = setInterval(() => {
      setSecondsAgo(Math.floor((Date.now() - dataUpdatedAt) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [dataUpdatedAt]);

  useEffect(() => {
    if (!data) return;
    const now = formatTime(new Date());

    const newOH = [...overheadHistoryRef.current, {
      time: now,
      "Overhead avg": data.latency.overhead?.avg_ms ?? undefined,
    }].slice(-MAX_HISTORY);
    overheadHistoryRef.current = newOH;
    setOverheadHistory([...newOH]);

    const newIF = [...inflightHistoryRef.current, {
      time: now,
      "In-flight": data.connection_pools.in_flight_requests ?? undefined,
    }].slice(-MAX_HISTORY);
    inflightHistoryRef.current = newIF;
    setInflightHistory([...newIF]);

    const newHTTP = [...httpHistoryRef.current, {
      time: now,
      "HTTP pool %": data.connection_pools.http.aiohttp_pct ?? undefined,
    }].slice(-MAX_HISTORY);
    httpHistoryRef.current = newHTTP;
    setHttpHistory([...newHTTP]);
  }, [dataUpdatedAt]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Spin size="large" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8">
        <Alert
          type="error"
          message="Failed to load performance data"
          description="Check that the proxy is running and your API key has admin access."
          showIcon
        />
      </div>
    );
  }

  const { debug_flags, workers, connection_pools, latency, per_model, issues } = data;
  const overheadHigh = latency.overhead_pct_of_total != null && latency.overhead_pct_of_total > 20;
  const workersLow = workers.num_workers < workers.cpu_count;

  // Delta vs ~5 min ago (oldest point in history)
  const oldestOverhead = overheadHistory.length > 1
    ? overheadHistory[0]["Overhead avg"]
    : null;
  const currentOverhead = latency.overhead?.avg_ms ?? null;
  const overheadDelta = oldestOverhead != null && currentOverhead != null
    ? Math.round((currentOverhead - oldestOverhead) * 10) / 10
    : null;

  return (
    <div style={{ width: "100%" }} className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between gap-6 mb-6">
        <div className="flex items-stretch gap-2 min-w-0">
          <div className="flex-shrink-0 flex items-center">
            <DashboardOutlined style={{ fontSize: "32px" }} />
          </div>
          <div className="flex-1 min-w-0 ml-1">
            <h3 className="text-sm font-semibold text-gray-900 mb-0.5 leading-tight">Performance</h3>
            <p className="text-xs text-gray-600 leading-tight">LiteLLM Proxy · Latency Diagnostics</p>
          </div>
        </div>
        <Tag color={secondsAgo < 15 ? "green" : "default"} className="text-xs flex-shrink-0">
          {dataUpdatedAt
            ? secondsAgo === 0 ? "Refreshed just now" : `Refreshed ${secondsAgo}s ago`
            : "Waiting for data…"}
        </Tag>
      </div>

      {/* ── Issues Detected ── */}
      <div className="mb-6">
        <Card>
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold text-gray-700">Issues Detected</p>
            {issues.length === 0 && latency.sample_count > 0 && (
              <span className="flex items-center gap-1 text-xs text-green-600">
                <CheckCircleOutlined /> All clear
              </span>
            )}
          </div>
          {issues.length === 0 ? (
            <p className="text-sm text-gray-400">
              {latency.sample_count === 0 ? "Send traffic to start analysis." : "No issues found — proxy looks healthy."}
            </p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400 border-b border-gray-100">
                  <th className="pb-2 font-medium w-4"></th>
                  <th className="pb-2 font-medium">Issue</th>
                  <th className="pb-2 font-medium">Suggested Fix</th>
                </tr>
              </thead>
              <tbody>
                {issues.map((issue, i) => (
                  <tr key={i} className="border-b border-gray-50 align-top">
                    <td className="py-2 pr-2">
                      <span className={`inline-block w-2 h-2 rounded-full mt-1.5 ${SEVERITY_DOT[issue.severity] ?? "bg-gray-400"}`} />
                    </td>
                    <td className="py-2 pr-4">
                      <p className="font-medium text-gray-800">{issue.title}</p>
                      <p className="text-xs text-gray-500 mt-0.5">{issue.description}</p>
                    </td>
                    <td className="py-2 min-w-[220px]">
                      {issue.fix_snippet ? (
                        <Collapse ghost size="small">
                          <Collapse.Panel header={<span className="text-xs text-blue-600">{issue.fix}</span>} key="1">
                            <pre className="text-xs font-mono bg-gray-50 border border-gray-200 rounded px-3 py-2 whitespace-pre-wrap">
                              {issue.fix_snippet}
                            </pre>
                          </Collapse.Panel>
                        </Collapse>
                      ) : (
                        <p className="text-xs text-gray-500">{issue.fix}</p>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </div>

      {/* ── Top metric cards ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="OVERHEAD AVG"
          value={latency.overhead ? `${latency.overhead.avg_ms} ms` : "—"}
          subtitle={
            latency.overhead
              ? `p50: ${latency.overhead.p50_ms}ms · p95: ${latency.overhead.p95_ms}ms${overheadDelta !== null ? ` · ${overheadDelta >= 0 ? "+" : ""}${overheadDelta}ms vs earlier` : ""}`
              : "No data yet"
          }
          valueColor={overheadHigh ? "text-orange-500" : "text-gray-900"}
        />
        <MetricCard
          label="OVERHEAD % OF TOTAL"
          value={latency.overhead_pct_of_total != null ? `${latency.overhead_pct_of_total}%` : "—"}
          subtitle="LiteLLM share of request time"
          valueColor={overheadHigh ? "text-orange-500" : "text-green-600"}
        />
        <MetricCard
          label="LLM API AVG"
          value={latency.llm_api ? `${latency.llm_api.avg_ms} ms` : "—"}
          subtitle={latency.llm_api ? `p50: ${latency.llm_api.p50_ms}ms · p95: ${latency.llm_api.p95_ms}ms` : "No data yet"}
        />
        <MetricCard
          label="TOTAL AVG"
          value={latency.total ? `${latency.total.avg_ms} ms` : "—"}
          subtitle={latency.total ? `p95: ${latency.total.p95_ms}ms · n=${latency.sample_count}` : "No data yet"}
        />
      </div>

      {/* ── Main 2-column layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Left: Overhead Over Time */}
        <div className="lg:col-span-2">
          <Card>
            <p className="text-sm font-semibold text-gray-700 mb-0.5">Overhead Over Time</p>
            <p className="text-xs text-gray-400 mb-3">LiteLLM-added latency · last 10 min · 10s resolution</p>

            {/* Plain-English summary */}
            {latency.sample_count > 0 && latency.overhead_histogram && (() => {
              const hist = latency.overhead_histogram;
              const total = hist.reduce((s, b) => s + b.count, 0);
              const under50 = hist.filter(b => ["0-5","5-10","10-25","25-50"].includes(b.bucket)).reduce((s,b) => s+b.count, 0);
              const pct = total > 0 ? Math.round(under50 / total * 100) : 0;
              const worstBucket = [...hist].reverse().find(b => b.count > 0);
              return (
                <div className="flex items-center gap-6 mb-3 px-1">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-gray-900">{total}</p>
                    <p className="text-xs text-gray-400">total requests</p>
                  </div>
                  <div className="text-center">
                    <p className={`text-2xl font-bold ${pct >= 90 ? "text-green-600" : pct >= 70 ? "text-amber-500" : "text-red-500"}`}>{pct}%</p>
                    <p className="text-xs text-gray-400">under 50ms overhead</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-gray-900">{latency.overhead?.p95_ms ?? "—"}ms</p>
                    <p className="text-xs text-gray-400">p95 overhead</p>
                  </div>
                  {worstBucket && worstBucket.bucket !== "0-5" && (
                    <div className="text-center">
                      <p className="text-2xl font-bold text-orange-500">{worstBucket.count}</p>
                      <p className="text-xs text-gray-400">requests &gt; {worstBucket.bucket.split("-")[0]}ms</p>
                    </div>
                  )}
                </div>
              );
            })()}

            {latency.sample_count === 0 ? (
              <div className="flex items-center justify-center h-40 text-gray-400 text-sm">
                No requests yet — send traffic through the proxy to see data.
              </div>
            ) : (
              <LineChart
                data={overheadHistory}
                index="time"
                categories={["Overhead avg"]}
                colors={["blue"]}
                valueFormatter={(v) => `${v} ms`}
                yAxisWidth={52}
                showLegend={false}
                showAnimation={false}
                className="h-40"
              />
            )}
          </Card>
        </div>

        {/* Right: Worker Provisioning + Connections */}
        <div className="flex flex-col gap-4">
          <Card>
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm font-semibold text-gray-700">WORKER PROVISIONING</p>
              {workersLow && <Tag color="orange" className="text-xs">Under-provisioned</Tag>}
            </div>
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <p className="text-xs text-gray-500">CPU Cores</p>
                <p className="text-2xl font-bold text-gray-900">{workers.cpu_count}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Workers</p>
                <p className={`text-2xl font-bold ${workersLow ? "text-orange-500" : "text-gray-900"}`}>
                  {workers.num_workers}
                </p>
              </div>
            </div>
            {workers.cpu_percent != null && (
              <div className="mb-3">
                <p className="text-xs text-gray-500">CPU Usage</p>
                <p className={`text-lg font-semibold ${workers.cpu_percent > 80 ? "text-red-500" : workers.cpu_percent > 60 ? "text-orange-500" : "text-green-600"}`}>
                  {workers.cpu_percent}%
                </p>
              </div>
            )}
            <div className="text-xs text-gray-500 mb-1">
              Workers / CPU ratio
              <span className="float-right">{workers.num_workers}/{workers.cpu_count}</span>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-1.5 mb-2">
              <div
                className={`h-1.5 rounded-full ${workersLow ? "bg-orange-400" : "bg-green-500"}`}
                style={{ width: `${Math.min((workers.num_workers / workers.cpu_count) * 100, 100)}%` }}
              />
            </div>
            {workersLow && (
              <p className="text-xs text-orange-600">Recommended: {2 * workers.cpu_count + 1} workers (2× CPU + 1)</p>
            )}
          </Card>

          <Card className="flex-1">
            <p className="text-sm font-semibold text-gray-700 mb-3">CONNECTIONS</p>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500 mb-0.5">Database</p>
                  <span className={`text-xs font-medium ${connection_pools.db.connected ? "text-green-600" : "text-red-500"}`}>
                    {connection_pools.db.connected ? "● Connected" : "● Disconnected"}
                  </span>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-500 mb-0.5">Pool limit</p>
                  <p className="text-lg font-bold text-gray-900">{connection_pools.db.pool_limit}</p>
                </div>
              </div>
              <div className="text-xs text-gray-400 -mt-1">
                Timeout: {connection_pools.db.pool_timeout_seconds}s
              </div>
              <div className="border-t border-gray-100 pt-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-gray-500 mb-0.5">Redis</p>
                    <span className={`text-xs font-medium ${connection_pools.redis.enabled ? "text-green-600" : "text-gray-400"}`}>
                      {connection_pools.redis.enabled ? "● Connected" : "● Not configured"}
                    </span>
                  </div>
                  {connection_pools.redis.enabled && (
                    <div className="text-right">
                      <p className="text-xs text-gray-500 mb-0.5">Max connections</p>
                      <p className="text-lg font-bold text-gray-900">{connection_pools.redis.max_connections ?? "∞"}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* ── Bottom row: In-Flight + HTTP pool ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <div className="flex items-center justify-between mb-0.5">
            <p className="text-sm font-semibold text-gray-700">In-Flight Requests</p>
            <span className="text-sm font-bold text-gray-700">
              {connection_pools.in_flight_requests ?? "—"}
              <span className="text-xs font-normal text-gray-400"> asyncio tasks</span>
            </span>
          </div>
          <p className="text-xs text-gray-400 mb-4">
            Concurrent asyncio tasks (proxy-wide) · last 10 min
          </p>
          <LineChart
            data={inflightHistory}
            index="time"
            categories={["In-flight"]}
            colors={["blue"]}
            yAxisWidth={40}
            showLegend={false}
            showAnimation={false}
            className="h-40"
          />
        </Card>

        <Card>
          <div className="flex items-center justify-between mb-0.5">
            <p className="text-sm font-semibold text-gray-700">HTTP Client Pool Utilization</p>
            <span className="text-sm font-bold text-gray-700">
              {connection_pools.http.aiohttp_active ?? "—"}
              <span className="text-xs font-normal text-gray-400">
                {" "}/ {connection_pools.http.aiohttp_limit} limit
                {connection_pools.http.aiohttp_pct != null && ` · ${connection_pools.http.aiohttp_pct}%`}
              </span>
            </span>
          </div>
          <p className="text-xs text-gray-400 mb-4">
            aiohttp active connections ÷ pool limit · amber line = 80% · last 10 min
          </p>
          <LineChart
            data={httpHistory}
            index="time"
            categories={["HTTP pool %"]}
            colors={["blue"]}
            valueFormatter={(v) => `${v}%`}
            yAxisWidth={44}
            showLegend={false}
            showAnimation={false}
            className="h-40"
            referenceLine={{ value: 80, label: "80%", color: "amber" }}
          />
        </Card>
      </div>

      {/* ── Per-model breakdown ── */}
      {per_model.length > 0 && (
        <div className="mb-6">
          <Card>
            <p className="text-sm font-semibold text-gray-700 mb-0.5">Per-Model Overhead</p>
            <p className="text-xs text-gray-400 mb-4">Sorted by overhead avg · last {latency.sample_count} requests</p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-400 border-b border-gray-100">
                    <th className="pb-2 font-medium">Model</th>
                    <th className="pb-2 font-medium text-right">Overhead avg</th>
                    <th className="pb-2 font-medium text-right">Overhead p95</th>
                    <th className="pb-2 font-medium text-right">LLM API avg</th>
                    <th className="pb-2 font-medium text-right">Total avg</th>
                    <th className="pb-2 font-medium text-right">Requests</th>
                  </tr>
                </thead>
                <tbody>
                  {per_model.map((row) => {
                    const overheadPct = row.overhead && row.total
                      ? Math.round((row.overhead.avg_ms / row.total.avg_ms) * 100)
                      : null;
                    return (
                      <tr key={row.model} className="border-b border-gray-50 hover:bg-gray-50">
                        <td className="py-2 font-mono text-xs text-gray-700 max-w-[200px] truncate pr-4">
                          {row.model}
                        </td>
                        <td className="py-2 text-right">
                          <span className={overheadPct != null && overheadPct > 20 ? "text-orange-500 font-semibold" : "text-gray-700"}>
                            {row.overhead ? `${row.overhead.avg_ms}ms` : "—"}
                          </span>
                          {overheadPct != null && (
                            <span className="text-xs text-gray-400 ml-1">({overheadPct}%)</span>
                          )}
                        </td>
                        <td className="py-2 text-right text-gray-600">
                          {row.overhead ? `${row.overhead.p95_ms}ms` : "—"}
                        </td>
                        <td className="py-2 text-right text-gray-600">
                          {row.llm_api ? `${row.llm_api.avg_ms}ms` : "—"}
                        </td>
                        <td className="py-2 text-right text-gray-600">
                          {row.total ? `${row.total.avg_ms}ms` : "—"}
                        </td>
                        <td className="py-2 text-right text-gray-400 text-xs">{row.sample_count}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Config summary ── */}
      <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Configuration Summary</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-xs text-gray-400 mb-1">Log Level</p>
            <Tag color={debug_flags.is_detailed_debug ? "red" : "green"}>{debug_flags.log_level}</Tag>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">Detailed Timing</p>
            <Tag color={debug_flags.detailed_timing_enabled ? "green" : "default"}>
              {debug_flags.detailed_timing_enabled ? "Enabled" : "Disabled"}
            </Tag>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">Workers / CPU</p>
            <span className="font-semibold text-gray-700">{workers.num_workers} / {workers.cpu_count}</span>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">Sample Count</p>
            <span className="font-semibold text-gray-700">{latency.sample_count}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
