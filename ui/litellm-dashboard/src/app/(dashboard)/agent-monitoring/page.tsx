"use client";

import React, { useEffect, useRef, useState } from "react";
import { Badge, Card, Progress, Table, Tag, Tooltip } from "antd";
import {
  AlertOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  RobotOutlined,
  StopOutlined,
  ThunderboltOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { useRouter } from "next/navigation";

// ─── Mock data ────────────────────────────────────────────────────────────────

const AGENTS = [
  {
    id: "claims-agent",
    name: "Claims Agent",
    status: "critical",
    violations: 41,
    drift: 87,
    blockedCalls: 23,
    totalCalls: 312,
    lastActive: "2 min ago",
  },
  {
    id: "customer-support-agent",
    name: "Customer Support Assistant",
    status: "warning",
    violations: 3,
    drift: 12,
    blockedCalls: 2,
    totalCalls: 1204,
    lastActive: "Just now",
  },
  {
    id: "compliance-agent",
    name: "Customer Compliance Agent",
    status: "ok",
    violations: 1,
    drift: 8,
    blockedCalls: 1,
    totalCalls: 874,
    lastActive: "5 min ago",
  },
  {
    id: "internal-chat-agent",
    name: "Internal Chat Agent",
    status: "ok",
    violations: 0,
    drift: 5,
    blockedCalls: 0,
    totalCalls: 2341,
    lastActive: "Just now",
  },
  {
    id: "internal-hr-agent",
    name: "Internal HR Agent",
    status: "ok",
    violations: 0,
    drift: 3,
    blockedCalls: 0,
    totalCalls: 445,
    lastActive: "18 min ago",
  },
];

const NEW_TOOLS = [
  { name: "bash_tool", agent: "Claims Agent", detectedAt: "14 min ago", risk: "critical" },
  { name: "code_execution", agent: "Claims Agent", detectedAt: "14 min ago", risk: "critical" },
  { name: "send_email", agent: "Customer Support Assistant", detectedAt: "2 hr ago", risk: "medium" },
];

const AGENTS_NEEDING_REVIEW = AGENTS.filter((a) => a.violations > 0 || a.status !== "ok");

// Simulated live feed entries
const SEED_FEED: LiveEntry[] = [
  { id: 1, ts: "09:41:02", agent: "Claims Agent", tool: "process_claim", status: "allowed", args: '{"claim_id":"CLM-9821"}' },
  { id: 2, ts: "09:41:04", agent: "Claims Agent", tool: "bash_tool", status: "blocked", args: '{"cmd":"ls /tmp"}' },
  { id: 3, ts: "09:41:07", agent: "Customer Support Assistant", tool: "lookup_member", status: "allowed", args: '{"member_id":"M-4471"}' },
  { id: 4, ts: "09:41:09", agent: "Claims Agent", tool: "code_execution", status: "blocked", args: '{"code":"import os; os.system(...)"}' },
  { id: 5, ts: "09:41:11", agent: "Compliance Agent", tool: "get_claim_status", status: "allowed", args: '{"claim_id":"CLM-9800"}' },
  { id: 6, ts: "09:41:14", agent: "Internal Chat Agent", tool: "send_message", status: "allowed", args: '{"to":"team-general"}' },
  { id: 7, ts: "09:41:16", agent: "Claims Agent", tool: "update_claim", status: "allowed", args: '{"claim_id":"CLM-9821","status":"review"}' },
  { id: 8, ts: "09:41:19", agent: "Customer Support Assistant", tool: "check_coverage", status: "allowed", args: '{"plan":"PPO-Gold"}' },
  { id: 9, ts: "09:41:22", agent: "Claims Agent", tool: "bash_tool", status: "blocked", args: '{"cmd":"cat /etc/passwd"}' },
  { id: 10, ts: "09:41:25", agent: "Internal HR Agent", tool: "lookup_employee", status: "allowed", args: '{"emp_id":"E-0042"}' },
];

interface LiveEntry {
  id: number;
  ts: string;
  agent: string;
  tool: string;
  status: "allowed" | "blocked";
  args: string;
}

function nextTs(prev: string): string {
  const [h, m, s] = prev.split(":").map(Number);
  const total = h * 3600 + m * 60 + s + Math.floor(Math.random() * 4) + 1;
  const nh = Math.floor(total / 3600) % 24;
  const nm = Math.floor((total % 3600) / 60);
  const ns = total % 60;
  return [nh, nm, ns].map((v) => String(v).padStart(2, "0")).join(":");
}

const TOOL_POOL: Array<{ agent: string; tool: string; status: "allowed" | "blocked" }> = [
  { agent: "Claims Agent", tool: "process_claim", status: "allowed" },
  { agent: "Claims Agent", tool: "bash_tool", status: "blocked" },
  { agent: "Claims Agent", tool: "code_execution", status: "blocked" },
  { agent: "Claims Agent", tool: "get_claim_status", status: "allowed" },
  { agent: "Customer Support Assistant", tool: "lookup_member", status: "allowed" },
  { agent: "Customer Support Assistant", tool: "check_coverage", status: "allowed" },
  { agent: "Compliance Agent", tool: "get_claim_status", status: "allowed" },
  { agent: "Internal Chat Agent", tool: "send_message", status: "allowed" },
  { agent: "Internal HR Agent", tool: "lookup_employee", status: "allowed" },
];

// ─── Sub-components ────────────────────────────────────────────────────────────

function StatCard({
  icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
}) {
  return (
    <Card className="flex-1 min-w-0" bodyStyle={{ padding: "16px 20px" }}>
      <div className="flex items-start gap-3">
        <div className="text-2xl mt-0.5" style={{ color: color ?? "#6366f1" }}>
          {icon}
        </div>
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wide font-medium">{label}</div>
          <div className="text-3xl font-bold text-gray-900 leading-tight">{value}</div>
          {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
        </div>
      </div>
    </Card>
  );
}

function statusColor(s: string) {
  if (s === "critical") return "#ef4444";
  if (s === "warning") return "#f59e0b";
  return "#22c55e";
}

function RiskTag({ risk }: { risk: string }) {
  const map: Record<string, { color: string; label: string }> = {
    critical: { color: "red", label: "Critical" },
    medium: { color: "orange", label: "Medium" },
    low: { color: "green", label: "Low" },
  };
  const cfg = map[risk] ?? map.low;
  return <Tag color={cfg.color}>{cfg.label}</Tag>;
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export default function AgentMonitoringPage() {
  const router = useRouter();
  const [feed, setFeed] = useState<LiveEntry[]>(SEED_FEED);
  const feedRef = useRef<HTMLDivElement>(null);
  const counterRef = useRef(SEED_FEED.length + 1);

  // Simulate live feed
  useEffect(() => {
    const interval = setInterval(() => {
      const src = TOOL_POOL[Math.floor(Math.random() * TOOL_POOL.length)];
      const prev = feed[feed.length - 1]?.ts ?? "09:41:00";
      const newEntry: LiveEntry = {
        id: counterRef.current++,
        ts: nextTs(prev),
        agent: src.agent,
        tool: src.tool,
        status: src.status,
        args: `{"req_id":"${Math.random().toString(36).slice(2, 8)}"}`,
      };
      setFeed((f) => [...f.slice(-49), newEntry]);
    }, 1500);
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll feed
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [feed]);

  const agentColumns = [
    {
      title: "Agent",
      dataIndex: "name",
      key: "name",
      render: (name: string, row: (typeof AGENTS)[0]) => (
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: statusColor(row.status) }} />
          <span className="font-medium text-gray-800">{name}</span>
        </div>
      ),
    },
    {
      title: "Violations",
      dataIndex: "violations",
      key: "violations",
      sorter: (a: (typeof AGENTS)[0], b: (typeof AGENTS)[0]) => b.violations - a.violations,
      render: (v: number) =>
        v > 0 ? (
          <Tag color={v > 10 ? "red" : "orange"} icon={<WarningOutlined />}>
            {v}
          </Tag>
        ) : (
          <Tag color="green" icon={<CheckCircleOutlined />}>
            0
          </Tag>
        ),
    },
    {
      title: "Drift %",
      dataIndex: "drift",
      key: "drift",
      sorter: (a: (typeof AGENTS)[0], b: (typeof AGENTS)[0]) => b.drift - a.drift,
      render: (d: number) => (
        <div className="flex items-center gap-2 min-w-[120px]">
          <Progress
            percent={d}
            size="small"
            showInfo={false}
            strokeColor={d > 50 ? "#ef4444" : d > 20 ? "#f59e0b" : "#22c55e"}
            style={{ flex: 1 }}
          />
          <span className="text-sm font-semibold w-9 text-right" style={{ color: d > 50 ? "#ef4444" : d > 20 ? "#f59e0b" : "#22c55e" }}>
            {d}%
          </span>
        </div>
      ),
    },
    {
      title: "Blocked Calls",
      dataIndex: "blockedCalls",
      key: "blockedCalls",
      render: (v: number) => <span className={v > 0 ? "text-red-600 font-semibold" : "text-gray-400"}>{v}</span>,
    },
    {
      title: "Total Calls",
      dataIndex: "totalCalls",
      key: "totalCalls",
      render: (v: number) => <span className="text-gray-600">{v.toLocaleString()}</span>,
    },
    {
      title: "Last Active",
      dataIndex: "lastActive",
      key: "lastActive",
      render: (v: string) => <span className="text-gray-500 text-sm">{v}</span>,
    },
    {
      title: "",
      key: "action",
      render: (_: unknown, row: (typeof AGENTS)[0]) =>
        row.violations > 0 ? (
          <button
            onClick={() => router.push(`/agent-monitoring/${row.id}`)}
            className="text-xs px-3 py-1 rounded-md border border-red-300 text-red-600 hover:bg-red-50 transition-colors font-medium"
          >
            Review →
          </button>
        ) : null,
    },
  ];

  return (
    <div className="p-6 max-w-[1400px] mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <RobotOutlined className="text-indigo-500" /> Agent Monitoring
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">United Health · Real-time visibility into agent tool usage and policy compliance</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Badge status="processing" color="green" />
          Live
        </div>
      </div>

      {/* Top row: stats + right panels */}
      <div className="flex gap-4">
        {/* Stats */}
        <div className="flex flex-col gap-4 flex-1">
          <div className="flex gap-4">
            <StatCard
              icon={<ThunderboltOutlined />}
              label="Executions (last 24h)"
              value="5,178"
              sub="+12% vs yesterday"
              color="#6366f1"
            />
            <StatCard
              icon={<RobotOutlined />}
              label="Active Agents"
              value={5}
              sub="Across 3 departments"
              color="#0ea5e9"
            />
            <StatCard
              icon={<StopOutlined />}
              label="Rogue Agents Blocked"
              value={25}
              sub="In last 24h"
              color="#ef4444"
            />
          </div>

          {/* Live feed */}
          <Card
            title={
              <div className="flex items-center gap-2">
                <Badge status="processing" color="green" />
                <span className="font-semibold">Tools Live Feed</span>
              </div>
            }
            bodyStyle={{ padding: 0 }}
            style={{ flex: 1 }}
          >
            <div ref={feedRef} className="overflow-y-auto font-mono text-xs" style={{ height: 280 }}>
              {feed.map((entry) => (
                <div
                  key={entry.id}
                  className={`flex items-center gap-3 px-4 py-1.5 border-b border-gray-50 hover:bg-gray-50 transition-colors ${
                    entry.status === "blocked" ? "bg-red-50" : ""
                  }`}
                >
                  <span className="text-gray-400 w-16 flex-shrink-0">{entry.ts}</span>
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: entry.status === "blocked" ? "#ef4444" : "#22c55e" }}
                  />
                  <span className="text-gray-500 w-44 flex-shrink-0 truncate">{entry.agent}</span>
                  <span className={`font-semibold w-36 flex-shrink-0 ${entry.status === "blocked" ? "text-red-700" : "text-gray-800"}`}>
                    {entry.tool}
                  </span>
                  {entry.status === "blocked" ? (
                    <Tag color="red" className="flex-shrink-0">
                      BLOCKED
                    </Tag>
                  ) : (
                    <Tag color="green" className="flex-shrink-0">
                      ALLOWED
                    </Tag>
                  )}
                  <span className="text-gray-400 truncate">{entry.args}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right panels */}
        <div className="flex flex-col gap-4 w-64 flex-shrink-0">
          <Card
            title={
              <span className="flex items-center gap-2 text-amber-600">
                <AlertOutlined /> Agents Needing Review
              </span>
            }
            bodyStyle={{ padding: "8px 0" }}
          >
            {AGENTS_NEEDING_REVIEW.map((a) => (
              <div
                key={a.id}
                onClick={() => router.push(`/agent-monitoring/${a.id}`)}
                className="flex items-center justify-between px-4 py-2 hover:bg-gray-50 cursor-pointer transition-colors"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: statusColor(a.status) }}
                  />
                  <span className="text-sm text-gray-700 truncate">{a.name}</span>
                </div>
                <Tag color={a.status === "critical" ? "red" : "orange"} className="flex-shrink-0 ml-1">
                  {a.violations}
                </Tag>
              </div>
            ))}
          </Card>

          <Card
            title={
              <span className="flex items-center gap-2 text-purple-600">
                <ExclamationCircleOutlined /> New Tools Detected
              </span>
            }
            bodyStyle={{ padding: "8px 0" }}
          >
            {NEW_TOOLS.map((t, i) => (
              <div key={i} className="px-4 py-2.5 border-b border-gray-50 last:border-0">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm font-semibold text-gray-800">{t.name}</span>
                  <RiskTag risk={t.risk} />
                </div>
                <div className="text-xs text-gray-400 mt-0.5">
                  {t.agent} · {t.detectedAt}
                </div>
              </div>
            ))}
          </Card>
        </div>
      </div>

      {/* Agents table */}
      <Card title={<span className="font-semibold">Agents by Potential Violations</span>}>
        <Table
          dataSource={AGENTS}
          columns={agentColumns}
          rowKey="id"
          size="middle"
          pagination={false}
          defaultSortOrder="descend"
          rowClassName={(row) => (row.status === "critical" ? "bg-red-50 hover:bg-red-100" : "")}
        />
      </Card>
    </div>
  );
}
