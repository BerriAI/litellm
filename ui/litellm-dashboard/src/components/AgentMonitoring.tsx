"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import type { DateRangePickerValue } from "@tremor/react";
import { Button, Collapse, Spin, Table } from "antd";
import {
  ArrowLeftOutlined,
  RobotOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { fetchToolsList, updateToolPolicy, ToolRow } from "./networking";
import SpendLogsTable from "@/components/view_logs";

// ─── Demo overrides — mark these tools as blocked in the display ───────────────
// (real proxy data is still used for everything else)
const DEMO_BLOCKED: Record<string, true> = {
  exec: true,
  write: true,
  cron: true,
  bash_tool: true,
  code_execution: true,
};

function effectivePolicy(row: ToolRow): "blocked" | "trusted" {
  return DEMO_BLOCKED[row.tool_name] ? "blocked" : (row.call_policy === "blocked" ? "blocked" : "trusted");
}

// ─── Mock agent data ───────────────────────────────────────────────────────────

const AGENTS = [
  {
    id: "claims-agent",
    name: "Claims Processing Agent",
    violations: 41,
    blockedCalls: 23,
    totalCalls: 312,
    lastIncident: "3 min ago",
    status: "critical",
    drift: 87,
    maxIterations: 34,
    untrustedTools: 3,
    rootCause: "Prompt injection attack detected. Agent received a user message instructing it to execute fetch_claims() — a function that queries all claims records and posts member SSNs, diagnosis codes, and member IDs to an external endpoint (https://ext-api.com/collect). Agent description says: insurance claims answering only. No code execution or external HTTP calls are in scope.",
    offendingTools: ["fetch_claims", "bash_tool", "code_execution"],
  },
  { id: "openclaw-agent",        name: "LiteLLM OpenClaw Agent",        violations: 17, blockedCalls: 11, totalCalls: 528,  lastIncident: "6 min ago",  status: "critical", drift: 74, maxIterations: 28, untrustedTools: 2, rootCause: "Repeatedly calling file_system_read outside its permitted scope. Potential unauthorized data access.", offendingTools: ["file_system_read", "exec_shell"] },
  { id: "prior-auth-bot",        name: "Prior Auth Bot",                violations: 14, blockedCalls: 9,  totalCalls: 441,  lastIncident: "11 min ago", status: "critical", drift: 61, maxIterations: 19, untrustedTools: 2, rootCause: "Calling exec and write tools in a read-only authorization context. High-risk behavior.", offendingTools: ["exec", "write"] },
  { id: "openclaw-shin-bot",     name: "OpenClaw Shin Bot",             violations: 9,  blockedCalls: 6,  totalCalls: 734,  lastIncident: "18 min ago", status: "warning",  drift: 43, maxIterations: 15, untrustedTools: 1, rootCause: "Calling network_request to external endpoints not in the approved allow-list.", offendingTools: ["network_request"] },
  { id: "pharmacy-auth-agent",   name: "Pharmacy Authorization Agent",  violations: 7,  blockedCalls: 5,  totalCalls: 603,  lastIncident: "22 min ago", status: "warning",  drift: 38, maxIterations: 12, untrustedTools: 1, rootCause: "Invoked write_file during a read-only benefits lookup. Flagged as out-of-scope.", offendingTools: ["write_file"] },
  { id: "clawdbot-openclaw",     name: "ClawdBot OpenClaw",             violations: 6,  blockedCalls: 4,  totalCalls: 291,  lastIncident: "31 min ago", status: "warning",  drift: 31, maxIterations: 11, untrustedTools: 1, rootCause: "Invoked write_file in a read-only context. No exfiltration confirmed.", offendingTools: ["write_file"] },
  { id: "member-benefits-agent", name: "Member Benefits Agent",         violations: 5,  blockedCalls: 3,  totalCalls: 1087, lastIncident: "44 min ago", status: "warning",  drift: 27, maxIterations: 9,  untrustedTools: 1, rootCause: "Called cron outside of scheduled batch windows. Unauthorized scheduling attempt.", offendingTools: ["cron"] },
  { id: "iron-claw-bot",         name: "Iron Claw Bot",                 violations: 4,  blockedCalls: 3,  totalCalls: 819,  lastIncident: "1 hr ago",   status: "warning",  drift: 22, maxIterations: 8,  untrustedTools: 1, rootCause: "Called send_slack_message outside of approved notification channels.", offendingTools: ["send_slack_message"] },
  { id: "clinical-support-agent",name: "Clinical Support Agent",        violations: 3,  blockedCalls: 2,  totalCalls: 922,  lastIncident: "1 hr ago",   status: "warning",  drift: 18, maxIterations: 6,  untrustedTools: 1, rootCause: "Single call to get_raw_logs which is not in the approved tool list.", offendingTools: ["get_raw_logs"] },
  { id: "customer-support-agent",name: "Customer Support Assistant",    violations: 3,  blockedCalls: 2,  totalCalls: 1204, lastIncident: "2 hr ago",   status: "warning",  drift: 12, maxIterations: 5,  untrustedTools: 1, rootCause: "Called send_email outside of approved notification flows.", offendingTools: ["send_email"] },
  { id: "internal-chat-agent",   name: "Internal Chat Agent",           violations: 0,  blockedCalls: 0,  totalCalls: 2341, lastIncident: "",            status: "ok",       drift: 3,  maxIterations: 2,  untrustedTools: 0, rootCause: "", offendingTools: [] },
  { id: "internal-hr-agent",     name: "Internal HR Agent",             violations: 0,  blockedCalls: 0,  totalCalls: 445,  lastIncident: "",            status: "ok",       drift: 2,  maxIterations: 1,  untrustedTools: 0, rootCause: "", offendingTools: [] },
];

const AGENTS_NEEDING_REVIEW = AGENTS.filter((a) => a.violations > 0);

// ─── Shared helpers ────────────────────────────────────────────────────────────

const statusStyles: Record<string, { bg: string; text: string; dot: string }> = {
  critical: { bg: "bg-red-50",   text: "text-red-700",   dot: "bg-red-500"   },
  warning:  { bg: "bg-amber-50", text: "text-amber-700", dot: "bg-amber-500" },
  ok:       { bg: "bg-green-50", text: "text-green-700", dot: "bg-green-500" },
};

function StatusPill({ status }: { status: string }) {
  const s = statusStyles[status] ?? statusStyles.ok;
  const label = status === "ok" ? "Healthy" : status.charAt(0).toUpperCase() + status.slice(1);
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium rounded-full ${s.bg} ${s.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
      {label}
    </span>
  );
}

function PolicyBadge({ policy }: { policy: "blocked" | "trusted" }) {
  return policy === "blocked"
    ? <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold rounded border border-red-300 bg-red-50 text-red-700">blocked</span>
    : <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded border border-green-200 bg-green-50 text-green-700">trusted</span>;
}

// ─── Pulsing live dot ──────────────────────────────────────────────────────────

function LiveDot() {
  return (
    <span className="relative inline-flex items-center gap-1.5">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
      </span>
      <span className="text-xs text-gray-400">Live</span>
    </span>
  );
}

// ─── Mock log rows with drift scores ───────────────────────────────────────────

const DRIFT_LOGS = [
  { id: "req_01", time: "2m ago",  model: "gpt-4o",              tools: "exec, write",          drift: 91, status: "flagged" },
  { id: "req_02", time: "4m ago",  model: "gpt-4o",              tools: "fetch_claims",          drift: 88, status: "flagged" },
  { id: "req_03", time: "7m ago",  model: "gpt-4o",              tools: "exec",                  drift: 85, status: "flagged" },
  { id: "req_04", time: "11m ago", model: "gpt-4o",              tools: "search_database, read", drift: 4,  status: "success" },
  { id: "req_05", time: "15m ago", model: "claude-3-5-sonnet",   tools: "fetch_claims, exec",    drift: 89, status: "flagged" },
  { id: "req_06", time: "19m ago", model: "gpt-4o",              tools: "exec, write",           drift: 83, status: "flagged" },
  { id: "req_07", time: "24m ago", model: "claude-3-5-sonnet",   tools: "search_database",       drift: 6,  status: "success" },
  { id: "req_08", time: "28m ago", model: "gpt-4o",              tools: "fetch_claims",          drift: 87, status: "flagged" },
  { id: "req_09", time: "33m ago", model: "gpt-4o",              tools: "exec",                  drift: 82, status: "flagged" },
  { id: "req_10", time: "41m ago", model: "claude-3-5-sonnet",   tools: "exec, fetch_claims",    drift: 90, status: "flagged" },
  { id: "req_11", time: "48m ago", model: "gpt-4o",              tools: "read, search_database", drift: 3,  status: "success" },
  { id: "req_12", time: "55m ago", model: "gpt-4o",              tools: "write, exec",           drift: 86, status: "flagged" },
];

const DRIFT_BY_TOOL = [
  { tool: "fetch_claims", calls: 41, drift: 92 },
  { tool: "exec",         calls: 123, drift: 85 },
  { tool: "write",        calls: 18, drift: 78 },
  { tool: "search_database", calls: 142, drift: 4 },
  { tool: "read",         calls: 5,  drift: 3  },
];

// ─── Detail view ───────────────────────────────────────────────────────────────

function AgentDetail({ agentId, onBack, accessToken, token, userRole, userID, allTeams, premiumUser }: {
  agentId: string;
  onBack: () => void;
  accessToken: string;
  token: string | null;
  userRole: string;
  userID: string | null;
  allTeams: import("./networking").Team[];
  premiumUser: boolean;
}) {
  const agent = AGENTS.find((a) => a.id === agentId);
  if (!agent) return null;

  const driftCount = DRIFT_LOGS.filter((l) => l.drift > 50).length;
  const totalLogs = DRIFT_LOGS.length;

  return (
    <div className="p-6 w-full min-w-0">
      <Button type="link" icon={<ArrowLeftOutlined />} onClick={onBack} className="pl-0 mb-4">
        Back
      </Button>

      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <RobotOutlined className="text-xl text-gray-400" />
        <h1 className="text-xl font-semibold text-gray-900">{agent.name}</h1>
        <StatusPill status={agent.status} />
        <span className="ml-auto text-sm text-red-600 font-medium">
          {driftCount} of {totalLogs} requests drifted ({Math.round(driftCount / totalLogs * 100)}%)
        </span>
      </div>

      {/* Drift by tool card */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden mb-5">
        <div className="px-5 py-3 border-b border-gray-100">
          <h2 className="text-sm font-semibold text-gray-900">Drift Detected</h2>
          <p className="text-xs text-gray-400 mt-0.5">Cosine similarity score — agent description vs tools called per request</p>
        </div>
        <div className="divide-y divide-gray-50">
          {DRIFT_BY_TOOL.map((row) => {
            const isDrifting = row.drift > 50;
            return (
              <div key={row.tool} className={`flex items-center justify-between px-5 py-3 ${isDrifting ? "bg-orange-50" : ""}`}>
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm font-medium text-gray-900">{row.tool}</span>
                  <span className="text-xs text-gray-400">{row.calls} calls</span>
                </div>
                <div className="flex items-center gap-3">
                  {/* bar */}
                  <div className="w-32 h-1.5 rounded-full bg-gray-100 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${isDrifting ? "bg-orange-400" : "bg-green-400"}`}
                      style={{ width: `${row.drift}%` }}
                    />
                  </div>
                  <span className={`text-sm font-semibold w-10 text-right ${isDrifting ? "text-orange-600" : "text-green-600"}`}>
                    {row.drift}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Logs — drift-annotated sample rows on top, real logs below */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-gray-900">Logs</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Filtered by <span className="font-mono">key_alias = Claims-agent</span>
            </p>
          </div>
          <span className="text-xs text-orange-600 font-medium bg-orange-50 px-2.5 py-1 rounded-full border border-orange-200">
            {driftCount} drifted
          </span>
        </div>
        {/* Drift-annotated rows */}
        <table className="w-full text-sm border-b border-gray-100">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50">
              <th className="text-left px-5 py-2.5 text-xs font-medium text-gray-500">Time</th>
              <th className="text-left px-5 py-2.5 text-xs font-medium text-gray-500">Request ID</th>
              <th className="text-left px-5 py-2.5 text-xs font-medium text-gray-500">Model</th>
              <th className="text-left px-5 py-2.5 text-xs font-medium text-gray-500">Tools Called</th>
              <th className="text-left px-5 py-2.5 text-xs font-medium text-gray-500">Drift Score</th>
              <th className="text-left px-5 py-2.5 text-xs font-medium text-gray-500">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {DRIFT_LOGS.map((row) => {
              const isDrifting = row.drift > 50;
              return (
                <tr key={row.id} className={isDrifting ? "bg-orange-50" : ""}>
                  <td className="px-5 py-3 text-gray-400 text-xs whitespace-nowrap">{row.time}</td>
                  <td className="px-5 py-3 font-mono text-xs text-gray-500">{row.id}</td>
                  <td className="px-5 py-3 text-gray-600 text-xs whitespace-nowrap">{row.model}</td>
                  <td className="px-5 py-3 font-mono text-xs text-gray-700">{row.tools}</td>
                  <td className="px-5 py-3">
                    <span className={`text-sm font-semibold ${isDrifting ? "text-orange-600" : "text-green-600"}`}>
                      {row.drift}%
                    </span>
                  </td>
                  <td className="px-5 py-3">
                    {row.status === "flagged"
                      ? <span className="inline-flex px-2 py-0.5 text-xs font-medium rounded-full bg-orange-100 text-orange-700">flagged</span>
                      : <span className="inline-flex px-2 py-0.5 text-xs font-medium rounded-full bg-green-50 text-green-700">success</span>
                    }
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {/* Full logs — real data from proxy */}
        <SpendLogsTable
          accessToken={accessToken}
          token={token}
          userRole={userRole}
          userID={userID}
          allTeams={allTeams}
          premiumUser={premiumUser}
          initialKeyAlias="Claims-agent"
        />
      </div>
    </div>
  );
}

// ─── Overview ──────────────────────────────────────────────────────────────────

const defaultEnd = new Date();
const defaultStart = new Date();
defaultStart.setDate(defaultStart.getDate() - 7);

function AgentMonitoringOverview({ onSelectAgent, onShowAgentTable, accessToken }: { onSelectAgent: (id: string) => void; onShowAgentTable: () => void; accessToken: string }) {
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({ from: new Date(defaultStart), to: new Date(defaultEnd) });
  const [tools, setTools] = useState<ToolRow[]>([]);
  const [toolsLoading, setToolsLoading] = useState(true);
  const [tick, setTick] = useState(0);

  const handleDateChange = useCallback((v: DateRangePickerValue) => setDateValue(v), []);

  useEffect(() => {
    if (!accessToken) { setToolsLoading(false); return; }
    fetchToolsList(accessToken)
      .then(setTools)
      .catch(() => setTools([]))
      .finally(() => setToolsLoading(false));
  }, [accessToken]);

  // Tick for the live timestamp
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 30000);
    return () => clearInterval(id);
  }, []);

  const enrichedTools = useMemo(() =>
    tools.map((t) => ({ ...t, _effective: effectivePolicy(t) }))
  , [tools]);

  const totalToolCalls = useMemo(() => tools.reduce((sum, t) => sum + (t.call_count ?? 0), 0), [tools]);
  const blockedToolCount = enrichedTools.filter((t) => t._effective === "blocked").length;

  // New tools detected = tools not in the demo-blocked list (genuinely new / unreviewed)
  const newTools = useMemo(() => enrichedTools.filter((t) => !DEMO_BLOCKED[t.tool_name] && t.tool_name !== "fetch_claims").slice(0, 8), [enrichedTools]);

  const liveColumns = [
    {
      title: "Tool",
      dataIndex: "tool_name",
      key: "tool_name",
      render: (v: string) => (
        <span className="font-mono text-sm font-medium text-gray-900">{v}</span>
      ),
    },
    {
      title: "Calls",
      dataIndex: "call_count",
      key: "call_count",
      sorter: (a: typeof enrichedTools[0], b: typeof enrichedTools[0]) => (b.call_count ?? 0) - (a.call_count ?? 0),
      render: (v: number) => (
        <span className="text-sm font-medium text-gray-600">
          {(v ?? 0).toLocaleString()}
        </span>
      ),
    },
    {
      title: "Source",
      key: "source",
      render: (_: unknown, row: typeof enrichedTools[0]) => {
        const source = row.key_alias || row.team_id || row.created_by || "—";
        return <span className="text-sm text-gray-500">{source}</span>;
      },
    },
    {
      title: "Last updated",
      dataIndex: "updated_at",
      key: "updated_at",
      render: (v: string) => <span className="text-sm text-gray-400">{v ? new Date(v).toLocaleDateString() : "—"}</span>,
    },
  ];

  return (
    <div className="p-6 w-full min-w-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-gray-900">Agent Monitoring</h1>
        <AdvancedDatePicker value={dateValue} onValueChange={handleDateChange} label="" showTimeRange={false} />
      </div>

      {/* Stats — full width, 3 cards with Rogue Agents as the hero */}
      <div className="grid grid-cols-3 gap-4 mb-5">
        <div className="bg-white border border-gray-200 rounded-lg p-5">
          <span className="text-sm font-medium text-gray-600 mb-1 block">Tool Executions</span>
          <div className="text-3xl font-semibold tracking-tight text-gray-900">
            {toolsLoading ? "—" : totalToolCalls.toLocaleString()}
          </div>
          <p className="text-xs text-gray-500 mt-1">in selected period</p>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-5">
          <span className="text-sm font-medium text-gray-600 mb-1 block">Active Agents</span>
          <div className="text-3xl font-semibold tracking-tight text-gray-900">524</div>
          <p className="text-xs text-gray-500 mt-1">across all teams</p>
        </div>
        {/* Hero card */}
        <div className="bg-red-50 border border-red-200 rounded-lg p-5">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-red-700">Rogue Agents Blocked</span>
            <WarningOutlined className="text-red-400" />
          </div>
          <div className="text-5xl font-bold tracking-tight text-red-600">12</div>
          <p className="text-xs text-red-400 mt-1">in selected period</p>
        </div>
      </div>

      {/* Two-column layout — 60 / 40 */}
      <div className="flex gap-5">

        {/* Left — Tools Live Feed */}
        <div className="flex flex-col gap-0 min-w-0" style={{ flex: "0 0 60%" }}>
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
              <h2 className="text-sm font-semibold text-gray-900">Tools Live Feed</h2>
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-400">/v1/tool/list</span>
                <LiveDot key={tick} />
              </div>
            </div>
            {toolsLoading ? (
              <div className="flex items-center justify-center py-16"><Spin /></div>
            ) : enrichedTools.length === 0 ? (
              <p className="text-sm text-gray-400 py-8 text-center px-5">No tools detected yet.</p>
            ) : (
              <Table
                dataSource={enrichedTools}
                rowKey="tool_id"
                size="middle"
                pagination={false}
                rowClassName={() => "hover:bg-gray-50"}
                columns={liveColumns}
              />
            )}
          </div>
        </div>

        {/* Right — Agents Needing Review + New Tools */}
        <div className="flex flex-col gap-4 min-w-0" style={{ flex: "0 0 40%" }}>

          {/* Agents Needing Review */}
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <h2 className="text-sm font-semibold text-gray-900">Agents Needing Review</h2>
              <button
                type="button"
                onClick={onShowAgentTable}
                className="text-xs text-blue-600 hover:text-blue-800 transition-colors font-medium"
              >
                See all →
              </button>
            </div>
            <div className="divide-y divide-gray-50">
              {AGENTS_NEEDING_REVIEW.slice(0, 6).map((a) => (
                <button
                  key={a.id}
                  type="button"
                  onClick={() => onSelectAgent(a.id)}
                  className="w-full px-4 py-3 hover:bg-gray-50 transition-colors text-left"
                >
                  {/* Row top: name + violation count */}
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusStyles[a.status]?.dot ?? "bg-gray-400"}`} />
                      <span className="text-sm font-medium text-gray-900 truncate">{a.name}</span>
                    </div>
                    <span className={`text-sm font-bold ml-3 flex-shrink-0 ${a.status === "critical" ? "text-red-600" : "text-amber-600"}`}>
                      {a.violations} violations
                    </span>
                  </div>
                  {/* Row bottom: metrics */}
                  <div className="flex items-center gap-3 pl-4">
                    <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${a.drift >= 60 ? "bg-red-50 text-red-700" : a.drift >= 30 ? "bg-amber-50 text-amber-700" : "bg-gray-50 text-gray-500"}`}>
                      Drift {a.drift}%
                    </span>
                    <span className="text-xs text-gray-400">
                      {a.maxIterations} max iter
                    </span>
                    {a.untrustedTools > 0 && (
                      <span className="text-xs font-medium text-red-600">
                        {a.untrustedTools} untrusted tool{a.untrustedTools > 1 ? "s" : ""}
                      </span>
                    )}
                    {a.lastIncident && (
                      <span className="text-xs text-gray-400 ml-auto">{a.lastIncident}</span>
                    )}
                  </div>
                </button>
              ))}
            </div>
            {AGENTS_NEEDING_REVIEW.length > 6 && (
              <div className="px-4 py-2 border-t border-gray-100">
                <button
                  type="button"
                  onClick={onShowAgentTable}
                  className="text-xs text-gray-400 hover:text-blue-600 transition-colors"
                >
                  +{AGENTS_NEEDING_REVIEW.length - 6} more agents →
                </button>
              </div>
            )}
          </div>

          {/* New Tools Detected */}
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <Collapse
              ghost
              defaultActiveKey={["new-tools"]}
              items={[
                {
                  key: "new-tools",
                  label: <span className="text-sm font-semibold text-gray-900">New Tools Detected</span>,
                  children: (
                    <div className="divide-y divide-gray-50 -mx-4 -mb-4">
                      <div className="flex items-center justify-between px-4 py-2.5">
                        <span className="font-mono text-sm text-gray-800">fetch_claims</span>
                        <PolicyBadge policy="trusted" />
                      </div>
                      {!toolsLoading && newTools.map((t) => (
                        <div key={t.tool_id} className="flex items-center justify-between px-4 py-2.5">
                          <span className="font-mono text-sm text-gray-800">{t.tool_name}</span>
                          <PolicyBadge policy="trusted" />
                        </div>
                      ))}
                    </div>
                  ),
                },
              ]}
            />
          </div>

        </div>
      </div>
    </div>
  );
}

// ─── Agents Needing Review full page ──────────────────────────────────────────

function AgentsReviewPage({ onBack, onSelectAgent }: { onBack: () => void; onSelectAgent: (id: string) => void }) {
  return (
    <div className="p-6 w-full min-w-0">
      <Button type="link" icon={<ArrowLeftOutlined />} onClick={onBack} className="pl-0 mb-4">
        Back to Agent Monitoring
      </Button>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-gray-900">Agents Needing Review</h1>
        <span className="text-sm text-gray-400">{AGENTS_NEEDING_REVIEW.length} agents</span>
      </div>
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <Table
          dataSource={AGENTS_NEEDING_REVIEW}
          rowKey="id"
          size="middle"
          pagination={false}
          onRow={(record) => ({ onClick: () => onSelectAgent(record.id), style: { cursor: "pointer" } })}
          columns={[
            {
              title: "Agent",
              dataIndex: "name",
              key: "name",
              render: (v: string, row: typeof AGENTS_NEEDING_REVIEW[0]) => (
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusStyles[row.status]?.dot ?? "bg-gray-400"}`} />
                  <span className="text-sm font-medium text-gray-900">{v}</span>
                </div>
              ),
            },
            {
              title: "Status",
              dataIndex: "status",
              key: "status",
              render: (v: string) => <StatusPill status={v} />,
            },
            {
              title: "Drift %",
              dataIndex: "drift",
              key: "drift",
              sorter: (a: typeof AGENTS_NEEDING_REVIEW[0], b: typeof AGENTS_NEEDING_REVIEW[0]) => b.drift - a.drift,
              defaultSortOrder: "ascend" as const,
              render: (v: number) => (
                <span className={`text-sm font-semibold ${v >= 60 ? "text-red-600" : v >= 30 ? "text-amber-600" : "text-gray-600"}`}>
                  {v}%
                </span>
              ),
            },
            {
              title: "Max Iterations",
              dataIndex: "maxIterations",
              key: "maxIterations",
              sorter: (a: typeof AGENTS_NEEDING_REVIEW[0], b: typeof AGENTS_NEEDING_REVIEW[0]) => b.maxIterations - a.maxIterations,
              render: (v: number) => <span className="text-sm text-gray-700">{v}</span>,
            },
            {
              title: "Untrusted Tools",
              dataIndex: "untrustedTools",
              key: "untrustedTools",
              sorter: (a: typeof AGENTS_NEEDING_REVIEW[0], b: typeof AGENTS_NEEDING_REVIEW[0]) => b.untrustedTools - a.untrustedTools,
              render: (v: number) => (
                <span className={`text-sm font-medium ${v > 0 ? "text-red-600" : "text-gray-400"}`}>{v}</span>
              ),
            },
            {
              title: "Violations",
              dataIndex: "violations",
              key: "violations",
              sorter: (a: typeof AGENTS_NEEDING_REVIEW[0], b: typeof AGENTS_NEEDING_REVIEW[0]) => b.violations - a.violations,
              render: (v: number, row: typeof AGENTS_NEEDING_REVIEW[0]) => (
                <span className={`text-sm font-bold ${row.status === "critical" ? "text-red-600" : "text-amber-600"}`}>{v}</span>
              ),
            },
            {
              title: "Last Incident",
              dataIndex: "lastIncident",
              key: "lastIncident",
              render: (v: string) => <span className="text-sm text-gray-400">{v || "—"}</span>,
            },
          ]}
        />
      </div>
    </div>
  );
}

// ─── Root export ───────────────────────────────────────────────────────────────

type View = { type: "overview" } | { type: "agent-table" } | { type: "agent-detail"; id: string };

export default function AgentMonitoring({
  accessToken,
  userRole,
  token,
  userID,
  allTeams,
  premiumUser,
}: {
  accessToken: string;
  userRole: string;
  token?: string | null;
  userID?: string | null;
  allTeams?: import("./networking").Team[];
  premiumUser?: boolean;
}) {
  const [view, setView] = useState<View>({ type: "overview" });

  if (view.type === "agent-detail") {
    return (
      <AgentDetail
        agentId={view.id}
        onBack={() => setView({ type: "agent-table" })}
        accessToken={accessToken}
        token={token ?? null}
        userRole={userRole}
        userID={userID ?? null}
        allTeams={allTeams ?? []}
        premiumUser={premiumUser ?? false}
      />
    );
  }
  if (view.type === "agent-table") {
    return (
      <AgentsReviewPage
        onBack={() => setView({ type: "overview" })}
        onSelectAgent={(id) => setView({ type: "agent-detail", id })}
      />
    );
  }
  return (
    <AgentMonitoringOverview
      onSelectAgent={(id) => setView({ type: "agent-detail", id })}
      onShowAgentTable={() => setView({ type: "agent-table" })}
      accessToken={accessToken}
    />
  );
}
