"use client";

import React, { useCallback, useRef, useState, useEffect } from "react";
import { Collapse, Select, Spin, Tabs, Tag, Tooltip } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  DashboardOutlined,
  InfoCircleOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";
import { LineChart, Card } from "@tremor/react";
import {
  usePerformanceSummary,
} from "@/app/(dashboard)/hooks/performanceSummary/usePerformanceSummary";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchAvailableModels, ModelGroup } from "@/components/playground/llm_calls/fetch_models";
import { proxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";

const MAX_HISTORY = 60;

interface HistoryPoint {
  time: string;
  "Overhead avg"?: number;
  "In-flight"?: number;
  "HTTP pool %"?: number;
}

interface TestResult {
  status: number;
  overheadMs: number | null;   // raw float from x-litellm-overhead-duration-ms
  wallMs: number;              // client-side wall clock ms
  proxyTotalMs: number;        // proxy-measured total (x-litellm-response-duration-ms) or fallback to wallMs
  model: string;
  responseText: string;
  allHeaders: Record<string, string>;  // all response headers
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

const SEVERITY_DOT: Record<string, string> = {
  critical: "bg-red-500",
  warning: "bg-amber-400",
  info: "bg-blue-400",
};

async function runTestRequest(accessToken: string, model: string, messages: { role: string; content: string }[]): Promise<TestResult> {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/chat/completions` : `/chat/completions`;
  const authHeader = getGlobalLitellmHeaderName();
  const start = Date.now();
  const resp = await fetch(url, {
    method: "POST",
    headers: { [authHeader]: `Bearer ${accessToken}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, max_tokens: 100, stream: false }),
  });
  const wallMs = Date.now() - start;
  const responseText = await resp.text();
  // Capture ALL headers via forEach (most reliable cross-browser method for custom headers)
  const allHeaders: Record<string, string> = {};
  resp.headers.forEach((v, k) => { allHeaders[k] = v; });
  // Read overhead from captured map — forEach and get() can behave differently for exposed headers
  const overheadRaw = allHeaders["x-litellm-overhead-duration-ms"] ?? null;
  const overheadParsed = overheadRaw && overheadRaw !== "None" ? parseFloat(overheadRaw) : NaN;
  const responseDurationRaw = captured["x-litellm-response-duration-ms"] ?? null;
  const responseDurationParsed = responseDurationRaw && responseDurationRaw !== "None" ? parseFloat(responseDurationRaw) : NaN;
  // Prefer the proxy's measured total over wall clock (avoids network jitter inflating LLM API time)
  const proxyTotalMs = isNaN(responseDurationParsed) ? wallMs : Math.round(responseDurationParsed);
  return {
    status: resp.status,
    overheadMs: isNaN(overheadParsed) ? null : parseFloat(overheadRaw!),
    wallMs,
    proxyTotalMs,
    model,
    responseText,
    allHeaders,
  };
}

// OTEL-style trace span row
function WaterfallRow({ label, startMs, durationMs, totalMs, color, tooltip }: {
  label: React.ReactNode;
  startMs: number;
  durationMs: number;
  totalMs: number;
  color: string;
  tooltip?: string;
}) {
  const startPct = totalMs > 0 ? (startMs / totalMs) * 100 : 0;
  const widthPct = totalMs > 0 ? (durationMs / totalMs) * 100 : 0;
  const isWide = widthPct > 12;
  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50/40 transition-colors group">
      <td className="py-3 pr-4 text-sm text-gray-600 w-48 whitespace-nowrap">
        <span className="flex items-center gap-1.5">
          {label}
          {tooltip && (
            <Tooltip title={tooltip} placement="right">
              <InfoCircleOutlined className="text-gray-300 text-xs cursor-help group-hover:text-gray-400 transition-colors" />
            </Tooltip>
          )}
        </span>
      </td>
      <td className="py-3 pr-4">
        <div className="relative h-6 rounded overflow-hidden" style={{ background: "transparent" }}>
          {/* Timeline grid */}
          {[25, 50, 75].map((pct) => (
            <div key={pct} className="absolute top-0 h-full w-px bg-gray-100" style={{ left: `${pct}%` }} />
          ))}
          {/* Span bar */}
          <div
            className={`absolute top-0.5 h-5 rounded ${color} shadow-sm transition-all`}
            style={{ left: `${startPct}%`, width: `${Math.max(widthPct, 0.4)}%`, minWidth: "4px" }}
          >
            {isWide && (
              <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white/90 tabular-nums">
                {durationMs}ms
              </span>
            )}
          </div>
        </div>
      </td>
      <td className="py-3 text-sm text-gray-700 text-right w-20 tabular-nums font-mono">{durationMs}ms</td>
    </tr>
  );
}

export default function PerformanceDashboardView() {
  const { data, isLoading, error, dataUpdatedAt } = usePerformanceSummary();
  const { accessToken } = useAuthorized();

  const overheadHistoryRef = useRef<HistoryPoint[]>([]);
  const [overheadHistory, setOverheadHistory] = useState<HistoryPoint[]>([]);
  const inflightHistoryRef = useRef<HistoryPoint[]>([]);
  const [inflightHistory, setInflightHistory] = useState<HistoryPoint[]>([]);
  const httpHistoryRef = useRef<HistoryPoint[]>([]);
  const [httpHistory, setHttpHistory] = useState<HistoryPoint[]>([]);

  const [secondsAgo, setSecondsAgo] = useState(0);
  useEffect(() => {
    if (!dataUpdatedAt) return;
    setSecondsAgo(0);
    const interval = setInterval(() => setSecondsAgo(Math.floor((Date.now() - dataUpdatedAt) / 1000)), 1000);
    return () => clearInterval(interval);
  }, [dataUpdatedAt]);

  useEffect(() => {
    if (!data) return;
    const now = formatTime(new Date());
    const newOH = [...overheadHistoryRef.current, { time: now, "Overhead avg": data.latency.overhead?.avg_ms ?? undefined }].slice(-MAX_HISTORY);
    overheadHistoryRef.current = newOH; setOverheadHistory([...newOH]);
    const newIF = [...inflightHistoryRef.current, { time: now, "In-flight": data.connection_pools.in_flight_requests ?? undefined }].slice(-MAX_HISTORY);
    inflightHistoryRef.current = newIF; setInflightHistory([...newIF]);
    const newHTTP = [...httpHistoryRef.current, { time: now, "HTTP pool %": data.connection_pools.http.aiohttp_pct ?? undefined }].slice(-MAX_HISTORY);
    httpHistoryRef.current = newHTTP; setHttpHistory([...newHTTP]);
  }, [dataUpdatedAt]);

  // Test Request state
  const [models, setModels] = useState<ModelGroup[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
  const [testMessages, setTestMessages] = useState([{ role: "user", content: "Say hello in one word." }]);
  const [isTestLoading, setIsTestLoading] = useState(false);
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [testError, setTestError] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    fetchAvailableModels(accessToken).then((m) => {
      const chat = m.filter((x) => !x.mode || x.mode === "chat");
      const list = chat.length > 0 ? chat : m;
      setModels(list);
      if (list.length > 0 && !selectedModel) setSelectedModel(list[0].model_group);
    }).catch(() => {});
  }, [accessToken]);

  const handleRunTest = useCallback(async () => {
    if (!accessToken || !selectedModel) return;
    setIsTestLoading(true); setTestError(null); setTestResult(null);
    try {
      setTestResult(await runTestRequest(accessToken, selectedModel, testMessages));
    } catch (e: any) {
      setTestError(e.message ?? "Request failed");
    } finally {
      setIsTestLoading(false);
    }
  }, [accessToken, selectedModel, testMessages]);

  if (isLoading) return <div className="flex justify-center items-center h-64"><Spin size="large" /></div>;
  if (error || !data) return <div className="p-8 text-sm text-red-500">Failed to load performance data.</div>;

  const { debug_flags, workers, connection_pools, latency, issues } = data;
  const overheadHigh = latency.overhead_pct_of_total != null && latency.overhead_pct_of_total > 20;
  const workersLow = workers.num_workers < workers.cpu_count;

  const overviewTab = (
    <div>
      {/* Issues Detected */}
      <div className="mb-6">
        <Card>
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold text-gray-700">Issues Detected</p>
            {issues.length === 0 && latency.sample_count > 0 && (
              <span className="flex items-center gap-1 text-xs text-green-600"><CheckCircleOutlined /> All clear</span>
            )}
          </div>
          {issues.length === 0 ? (
            <p className="text-sm text-gray-400">{latency.sample_count === 0 ? "Send traffic to start analysis." : "No issues found — proxy looks healthy."}</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400 border-b border-gray-100">
                  <th className="pb-2 w-4"></th>
                  <th className="pb-2 font-medium">Issue</th>
                  <th className="pb-2 font-medium">Suggested Fix</th>
                </tr>
              </thead>
              <tbody>
                {issues.map((issue, i) => (
                  <tr key={i} className="border-b border-gray-50 align-top">
                    <td className="py-2 pr-2"><span className={`inline-block w-2 h-2 rounded-full mt-1.5 ${SEVERITY_DOT[issue.severity] ?? "bg-gray-400"}`} /></td>
                    <td className="py-2 pr-4">
                      <p className="font-medium text-gray-800">{issue.title}</p>
                      <p className="text-xs text-gray-500 mt-0.5">{issue.description}</p>
                    </td>
                    <td className="py-2 min-w-[200px]">
                      {issue.fix_snippet ? (
                        <Collapse ghost size="small">
                          <Collapse.Panel header={<span className="text-xs text-blue-600">{issue.fix}</span>} key="1">
                            <pre className="text-xs font-mono bg-gray-50 border border-gray-200 rounded px-3 py-2 whitespace-pre-wrap">{issue.fix_snippet}</pre>
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

      {/* Current Overhead | Worker Provisioning */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <p className="text-sm font-semibold text-gray-700 mb-4">Current Overhead</p>
          {latency.sample_count === 0 ? (
            <p className="text-sm text-gray-400">No data — send traffic through the proxy.</p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-y-4 mb-4">
                <div><p className="text-xs text-gray-500">Avg</p><p className={`text-2xl font-bold ${overheadHigh ? "text-orange-500" : "text-gray-900"}`}>{latency.overhead ? `${latency.overhead.avg_ms}ms` : "—"}</p></div>
                <div><p className="text-xs text-gray-500">p50</p><p className="text-2xl font-bold text-gray-900">{latency.overhead ? `${latency.overhead.p50_ms}ms` : "—"}</p></div>
                <div><p className="text-xs text-gray-500">p95</p><p className="text-2xl font-bold text-gray-900">{latency.overhead ? `${latency.overhead.p95_ms}ms` : "—"}</p></div>
                <div><p className="text-xs text-gray-500">% of total</p><p className={`text-2xl font-bold ${overheadHigh ? "text-orange-500" : "text-green-600"}`}>{latency.overhead_pct_of_total != null ? `${latency.overhead_pct_of_total}%` : "—"}</p></div>
              </div>
              <p className="text-xs text-gray-400 mb-2">Last {latency.sample_count} requests · LLM API avg: {latency.llm_api ? `${latency.llm_api.avg_ms}ms` : "—"} · Total avg: {latency.total ? `${latency.total.avg_ms}ms` : "—"}</p>
              <LineChart data={overheadHistory} index="time" categories={["Overhead avg"]} colors={["blue"]} valueFormatter={(v) => `${v}ms`} yAxisWidth={48} showLegend={false} showAnimation={false} className="h-28" />
            </>
          )}
        </Card>

        <Card>
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm font-semibold text-gray-700">Worker Provisioning</p>
            {workersLow && <Tag color="orange" className="text-xs">Under-provisioned</Tag>}
          </div>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div><p className="text-xs text-gray-500">CPU Cores</p><p className="text-2xl font-bold text-gray-900">{workers.cpu_count}</p></div>
            <div><p className="text-xs text-gray-500">Workers</p><p className={`text-2xl font-bold ${workersLow ? "text-orange-500" : "text-gray-900"}`}>{workers.num_workers}</p></div>
            <div>
              <p className="text-xs text-gray-500">CPU Usage</p>
              <p className={`text-2xl font-bold ${workers.cpu_percent == null ? "text-gray-400" : workers.cpu_percent > 80 ? "text-red-500" : workers.cpu_percent > 60 ? "text-orange-500" : "text-green-600"}`}>
                {workers.cpu_percent != null ? `${workers.cpu_percent}%` : "—"}
              </p>
            </div>
          </div>
          <div className="text-xs text-gray-500 mb-1">Workers / CPU <span className="float-right">{workers.num_workers}/{workers.cpu_count}</span></div>
          <div className="w-full bg-gray-100 rounded-full h-1.5">
            <div className={`h-1.5 rounded-full ${workersLow ? "bg-orange-400" : "bg-green-500"}`} style={{ width: `${Math.min((workers.num_workers / workers.cpu_count) * 100, 100)}%` }} />
          </div>
          {workersLow && <p className="text-xs text-orange-600 mt-2">Recommended: {2 * workers.cpu_count + 1} workers (2× CPU + 1)</p>}
        </Card>
      </div>

      {/* Proxy Settings */}
      <div className="mb-6">
        <Card>
          <p className="text-sm font-semibold text-gray-700 mb-3">Proxy Settings</p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-400 border-b border-gray-100">
                <th className="pb-2 font-medium">Setting</th>
                <th className="pb-2 font-medium">Current Value</th>
                <th className="pb-2 font-medium">Optimized for Performance</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-50">
                <td className="py-3 text-gray-700">Debug Mode</td>
                <td className="py-3"><Tag color={debug_flags.is_detailed_debug ? "red" : "default"} className="text-xs">{debug_flags.log_level}</Tag></td>
                <td className="py-3">{!debug_flags.is_detailed_debug ? <span className="flex items-center gap-1.5 text-green-600 text-xs font-medium"><CheckCircleOutlined /> Yes</span> : <span className="flex items-center gap-1.5 text-red-500 text-xs font-medium"><CloseCircleOutlined /> No — disable DEBUG logging in production</span>}</td>
              </tr>
              <tr className="border-b border-gray-50">
                <td className="py-3 text-gray-700">Detailed Timing Headers</td>
                <td className="py-3"><Tag color={debug_flags.detailed_timing_enabled ? "blue" : "default"} className="text-xs">{debug_flags.detailed_timing_enabled ? "Enabled" : "Disabled"}</Tag></td>
                <td className="py-3">{!debug_flags.detailed_timing_enabled ? <span className="flex items-center gap-1.5 text-green-600 text-xs font-medium"><CheckCircleOutlined /> Yes</span> : <span className="flex items-center gap-1.5 text-gray-500 text-xs font-medium"><CheckCircleOutlined className="text-gray-400" /> Minor overhead — safe to keep for debugging</span>}</td>
              </tr>
              <tr>
                <td className="py-3 text-gray-700">Worker Count</td>
                <td className="py-3"><span className="text-gray-700 font-medium">{workers.num_workers}</span><span className="text-gray-400 text-xs ml-1">/ {workers.cpu_count} CPU cores</span></td>
                <td className="py-3">{!workersLow ? <span className="flex items-center gap-1.5 text-green-600 text-xs font-medium"><CheckCircleOutlined /> Yes</span> : <span className="flex items-center gap-1.5 text-orange-500 text-xs font-medium"><CloseCircleOutlined /> No — recommend {2 * workers.cpu_count + 1} workers (2× CPU + 1)</span>}</td>
              </tr>
            </tbody>
          </table>
        </Card>
      </div>

      {/* Active Async Tasks | HTTP Pool */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="flex items-center justify-between mb-0.5">
            <p className="text-sm font-semibold text-gray-700">Active Async Tasks</p>
            <span className="text-sm font-bold text-gray-700">{connection_pools.in_flight_requests ?? "—"}<span className="text-xs font-normal text-gray-400"> tasks</span></span>
          </div>
          <p className="text-xs text-gray-400 mb-4">Total asyncio tasks running on this worker · last 10 min</p>
          <LineChart data={inflightHistory} index="time" categories={["In-flight"]} colors={["blue"]} yAxisWidth={40} showLegend={false} showAnimation={false} className="h-36" />
        </Card>
        <Card>
          <div className="flex items-center justify-between mb-0.5">
            <p className="text-sm font-semibold text-gray-700">HTTP Client Pool</p>
            <span className="text-sm font-bold text-gray-700">{connection_pools.http.aiohttp_active ?? "—"}<span className="text-xs font-normal text-gray-400"> / {connection_pools.http.aiohttp_limit}{connection_pools.http.aiohttp_pct != null && ` · ${connection_pools.http.aiohttp_pct}%`}</span></span>
          </div>
          <p className="text-xs text-gray-400 mb-4">aiohttp active / pool limit · last 10 min</p>
          <LineChart data={httpHistory} index="time" categories={["HTTP pool %"]} colors={["blue"]} valueFormatter={(v) => `${v}%`} yAxisWidth={44} showLegend={false} showAnimation={false} className="h-36" referenceLine={{ value: 80, label: "80%", color: "amber" }} />
        </Card>
      </div>
    </div>
  );

  // Use proxy-measured total for waterfall so LLM API bar = proxyTotal - overhead (no network noise)
  const waterfallTotal = testResult?.proxyTotalMs ?? 0;
  const llmApiMs = testResult ? Math.max(0, waterfallTotal - (testResult.overheadMs ?? 0)) : 0;
  const overheadPctTest = testResult && testResult.overheadMs != null && waterfallTotal > 0
    ? parseFloat(((testResult.overheadMs / waterfallTotal) * 100).toFixed(1))
    : null;
  // Display string: show 1 decimal for sub-10ms, otherwise round
  const overheadDisplay = testResult?.overheadMs != null
    ? testResult.overheadMs < 10
      ? `${testResult.overheadMs.toFixed(2)}ms`
      : `${Math.round(testResult.overheadMs)}ms`
    : null;

  let parsedContent = "";
  if (testResult) {
    try {
      const parsed = JSON.parse(testResult.responseText);
      parsedContent = parsed?.choices?.[0]?.message?.content ?? testResult.responseText;
    } catch {
      parsedContent = testResult.responseText;
    }
  }

  const testTab = (
    <div>
      <Card>
        {/* Controls */}
        <div className="flex items-end gap-3 mb-6">
          <div>
            <p className="text-xs text-gray-500 mb-1">Model</p>
            <Select
              value={selectedModel}
              placeholder="Select model"
              onChange={setSelectedModel}
              options={models.map((m) => ({ value: m.model_group, label: m.model_group }))}
              style={{ width: 260 }}
              showSearch
            />
          </div>
          <button
            onClick={handleRunTest}
            disabled={isTestLoading || !selectedModel}
            className={`flex items-center gap-2 px-4 py-1.5 rounded text-sm font-medium transition-colors ${isTestLoading || !selectedModel ? "bg-gray-100 text-gray-400 cursor-not-allowed" : "bg-blue-600 text-white hover:bg-blue-700 cursor-pointer"}`}
          >
            {isTestLoading ? <Spin size="small" /> : <PlayCircleOutlined />}
            {isTestLoading ? "Running…" : "Run Test"}
          </button>
        </div>

        {/* Message editor */}
        <div className="mb-6">
          <p className="text-xs text-gray-500 mb-2">Messages</p>
          {testMessages.map((msg, i) => (
            <div key={i} className="flex gap-2 mb-2 items-start">
              <select
                value={msg.role}
                onChange={(e) => setTestMessages((prev) => prev.map((m, j) => j === i ? { ...m, role: e.target.value } : m))}
                className="text-xs border border-gray-200 rounded px-2 py-1.5 text-gray-600 bg-white w-24 flex-shrink-0"
              >
                <option value="user">user</option>
                <option value="system">system</option>
                <option value="assistant">assistant</option>
              </select>
              <textarea
                value={msg.content}
                onChange={(e) => setTestMessages((prev) => prev.map((m, j) => j === i ? { ...m, content: e.target.value } : m))}
                rows={2}
                className="flex-1 text-sm border border-gray-200 rounded px-3 py-1.5 text-gray-700 resize-y font-mono"
              />
              {testMessages.length > 1 && (
                <button onClick={() => setTestMessages((prev) => prev.filter((_, j) => j !== i))} className="text-gray-300 hover:text-gray-500 text-xs pt-1.5">✕</button>
              )}
            </div>
          ))}
          <button
            onClick={() => setTestMessages((prev) => [...prev, { role: "user", content: "" }])}
            className="text-xs text-blue-500 hover:text-blue-700 mt-1"
          >
            + Add message
          </button>
        </div>

        {testError && <p className="text-xs text-red-500 mb-4">{testError}</p>}

        {testResult && (
          <>
            {/* Status bar */}
            <div className="flex items-center gap-0 mb-6 border border-gray-100 rounded-md overflow-hidden text-sm divide-x divide-gray-100">
              <div className="px-4 py-2.5 bg-gray-50">
                <span className={`font-semibold ${testResult.status === 200 ? "text-green-600" : "text-red-500"}`}>
                  {testResult.status === 200 ? "200 OK" : `${testResult.status} Error`}
                </span>
              </div>
              <div className="px-4 py-2.5 bg-white flex flex-col">
                <span className="text-[10px] text-gray-400 uppercase tracking-wide leading-none mb-0.5">Total</span>
                <span className="font-semibold text-gray-800 tabular-nums">{testResult.proxyTotalMs}ms</span>
              </div>
              <div className={`px-4 py-2.5 flex flex-col ${testResult.overheadMs === null ? "bg-amber-50" : overheadPctTest != null && overheadPctTest > 20 ? "bg-orange-50" : "bg-white"}`}>
                <span className="text-[10px] text-gray-400 uppercase tracking-wide leading-none mb-0.5">LiteLLM Overhead</span>
                {overheadDisplay !== null ? (
                  <span className={`font-semibold tabular-nums ${overheadPctTest != null && overheadPctTest > 20 ? "text-orange-500" : "text-blue-600"}`}>
                    {overheadDisplay}
                  </span>
                ) : (
                  <Tooltip title="x-litellm-overhead-duration-ms header not returned — upgrade to a recent proxy version">
                    <span className="text-amber-500 text-xs font-medium cursor-help">not available ⚠</span>
                  </Tooltip>
                )}
              </div>
              {overheadPctTest !== null && testResult.overheadMs !== null && (
                <div className={`px-4 py-2.5 flex flex-col ${overheadPctTest > 20 ? "bg-orange-50" : "bg-white"}`}>
                  <span className="text-[10px] text-gray-400 uppercase tracking-wide leading-none mb-0.5">Overhead %</span>
                  <span className={`font-semibold tabular-nums ${overheadPctTest > 20 ? "text-orange-500" : "text-blue-600"}`}>
                    {overheadPctTest}%
                  </span>
                </div>
              )}
              <div className="px-4 py-2.5 bg-white flex-1 flex items-center">
                <span className="text-gray-400 text-xs truncate">{testResult.model}</span>
              </div>
            </div>

            {/* OTEL-style trace waterfall */}
            <div className="mb-6">
              <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">Response Time Breakdown</p>

              {/* Time axis */}
              <div className="flex items-center mb-1">
                <div className="w-48 flex-shrink-0" />
                <div className="flex-1 pr-4 relative h-4">
                  {[0, 25, 50, 75, 100].map((pct) => (
                    <span
                      key={pct}
                      className="absolute text-[10px] text-gray-300 tabular-nums select-none"
                      style={{
                        left: `${pct}%`,
                        transform: pct === 0 ? "none" : pct === 100 ? "translateX(-100%)" : "translateX(-50%)",
                      }}
                    >
                      {Math.round((pct / 100) * waterfallTotal)}ms
                    </span>
                  ))}
                </div>
                <div className="w-20" />
              </div>

              {/* Span rows */}
              <div className="border border-gray-100 rounded-md overflow-hidden">
                <table className="w-full">
                  <tbody>
                    <WaterfallRow
                      label="LiteLLM Processing"
                      startMs={0}
                      durationMs={testResult.overheadMs ?? 0}
                      totalMs={waterfallTotal}
                      color={overheadPctTest != null && overheadPctTest > 20 ? "bg-orange-400" : "bg-blue-500"}
                      tooltip="Time LiteLLM spends on auth, routing, request transformation, and logging — excludes time waiting for the LLM API to respond."
                    />
                    <WaterfallRow
                      label="LLM API (waiting)"
                      startMs={testResult.overheadMs ?? 0}
                      durationMs={llmApiMs}
                      totalMs={waterfallTotal}
                      color="bg-teal-400"
                    />
                    {/* Total footer row */}
                    <tr className="bg-gray-50/50">
                      <td className="py-3 pr-4 text-sm font-semibold text-gray-700 w-48 pl-2">Total</td>
                      <td className="py-3 pr-4">
                        <div className="relative h-6 bg-gray-100 rounded">
                          {[25, 50, 75].map((pct) => (
                            <div key={pct} className="absolute top-0 h-full w-px bg-gray-200" style={{ left: `${pct}%` }} />
                          ))}
                          <div className="absolute top-0.5 left-0 h-5 w-full rounded bg-gradient-to-r from-blue-100 to-teal-100 opacity-70" />
                        </div>
                      </td>
                      <td className="py-3 text-sm font-semibold text-gray-900 text-right w-20 tabular-nums font-mono">{testResult.proxyTotalMs}ms</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="mt-2 text-xs text-gray-400">
                {overheadPctTest != null ? (
                  <span>
                    LiteLLM overhead:{" "}
                    <span className={`font-medium ${overheadPctTest > 20 ? "text-orange-500" : "text-blue-600"}`}>
                      {overheadDisplay} ({overheadPctTest}% of {testResult.proxyTotalMs}ms)
                    </span>
                    {overheadPctTest > 20 && <span className="text-orange-500"> — higher than expected</span>}
                  </span>
                ) : testResult.overheadMs === null ? (
                  <span className="text-amber-500">Overhead header not returned — check proxy version</span>
                ) : null}
              </div>
            </div>

            {/* Response content */}
            {parsedContent && (
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Response</p>
                <div className="bg-gray-50 border border-gray-100 rounded px-3 py-2.5 text-sm text-gray-700 font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {parsedContent}
                </div>
              </div>
            )}

            {/* Response headers */}
            {Object.keys(testResult.allHeaders).length > 0 && (
              <div className="mt-4">
                <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">Response Headers</p>
                <div className="border border-gray-100 rounded-md overflow-hidden">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-100">
                        <th className="py-1.5 px-3 text-left font-medium text-gray-400 w-64">Header</th>
                        <th className="py-1.5 px-3 text-left font-medium text-gray-400">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(testResult.allHeaders)
                        .sort(([a], [b]) => {
                          // x-litellm headers first
                          const aLitellm = a.startsWith("x-litellm");
                          const bLitellm = b.startsWith("x-litellm");
                          if (aLitellm && !bLitellm) return -1;
                          if (!aLitellm && bLitellm) return 1;
                          return a.localeCompare(b);
                        })
                        .map(([k, v]) => {
                          const isLitellm = k.startsWith("x-litellm");
                          const isOverhead = k === "x-litellm-overhead-duration-ms";
                          return (
                            <tr key={k} className={`border-b border-gray-50 ${isOverhead ? "bg-blue-50/60" : ""}`}>
                              <td className={`py-1.5 px-3 font-mono ${isLitellm ? "text-blue-600" : "text-gray-400"} ${isOverhead ? "font-semibold" : ""}`}>
                                {k}
                              </td>
                              <td className={`py-1.5 px-3 font-mono ${isLitellm ? "text-gray-700" : "text-gray-400"} ${isOverhead ? "font-semibold text-blue-700" : ""} break-all`}>
                                {v}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}

        {!testResult && !testError && !isTestLoading && (
          <p className="text-sm text-gray-400">Select a model and run a test to see latency breakdown.</p>
        )}
      </Card>
    </div>
  );

  return (
    <div style={{ width: "100%" }} className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <DashboardOutlined style={{ fontSize: "24px" }} />
          <div>
            <h3 className="text-sm font-semibold text-gray-900 leading-tight">Performance</h3>
            <p className="text-xs text-gray-500 leading-tight">LiteLLM Proxy · Latency Diagnostics</p>
          </div>
        </div>
        <Tag color={secondsAgo < 15 ? "green" : "default"} className="text-xs">
          {dataUpdatedAt ? (secondsAgo === 0 ? "just now" : `${secondsAgo}s ago`) : "waiting…"}
        </Tag>
      </div>

      <Tabs
        defaultActiveKey="overview"
        items={[
          { key: "overview", label: "Overview", children: overviewTab },
          { key: "test", label: "Test Request", children: testTab },
        ]}
      />
    </div>
  );
}
