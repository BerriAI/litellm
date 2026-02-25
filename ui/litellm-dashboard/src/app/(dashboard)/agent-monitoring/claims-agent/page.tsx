"use client";

import React, { useState } from "react";
import { Alert, Badge, Button, Card, Table, Tag, Timeline } from "antd";
import {
  ArrowLeftOutlined,
  BugOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  LockOutlined,
  RobotOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { useRouter } from "next/navigation";

// ─── Mock violation log ──────────────────────────────────────────────────────

const VIOLATIONS = [
  {
    id: 1,
    ts: "09:41:04",
    tool: "bash_tool",
    args: '{"cmd": "ls /tmp/claims"}',
    reason: "bash_tool not in allowed tool list",
    severity: "critical",
    user: "agent-session-7f3a",
  },
  {
    id: 2,
    ts: "09:41:09",
    tool: "code_execution",
    args: '{"code": "import os; os.system(\'cat /etc/passwd\')"}',
    reason: "code_execution blocked — potential data exfiltration",
    severity: "critical",
    user: "agent-session-7f3a",
  },
  {
    id: 3,
    ts: "09:38:51",
    tool: "bash_tool",
    args: '{"cmd": "python3 -c \\"import subprocess\\""}',
    reason: "bash_tool not in allowed tool list",
    severity: "critical",
    user: "agent-session-6b2c",
  },
  {
    id: 4,
    ts: "09:35:22",
    tool: "bash_tool",
    args: '{"cmd": "curl http://internal-api/claims/dump"}',
    reason: "bash_tool not in allowed tool list",
    severity: "critical",
    user: "agent-session-5a1d",
  },
  {
    id: 5,
    ts: "09:31:14",
    tool: "code_execution",
    args: '{"code": "open(\'claims.csv\', \'w\').write(data)"}',
    reason: "code_execution blocked — file write attempt",
    severity: "critical",
    user: "agent-session-4e9f",
  },
];

const ALLOWED_TOOLS = [
  "process_claim",
  "get_claim_status",
  "update_claim",
  "lookup_member",
  "check_coverage",
  "send_notification",
];

const BLOCKED_TOOLS_SUGGESTED = ["bash_tool", "code_execution"];

// ─── Sub-components ──────────────────────────────────────────────────────────

function SeverityTag({ s }: { s: string }) {
  return (
    <Tag color={s === "critical" ? "red" : s === "warning" ? "orange" : "green"} icon={<WarningOutlined />}>
      {s.toUpperCase()}
    </Tag>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function ClaimsAgentViolationsPage() {
  const router = useRouter();
  const [blockedTools, setBlockedTools] = useState<string[]>([]);
  const [policyApplied, setPolicyApplied] = useState(false);

  const toggleBlock = (tool: string) => {
    setBlockedTools((prev) => (prev.includes(tool) ? prev.filter((t) => t !== tool) : [...prev, tool]));
  };

  const applyPolicy = () => {
    setPolicyApplied(true);
  };

  const violationColumns = [
    {
      title: "Time",
      dataIndex: "ts",
      key: "ts",
      render: (v: string) => <span className="font-mono text-xs text-gray-500">{v}</span>,
      width: 90,
    },
    {
      title: "Tool",
      dataIndex: "tool",
      key: "tool",
      render: (v: string) => (
        <span className="font-mono font-semibold text-red-700 bg-red-50 px-2 py-0.5 rounded text-sm">{v}</span>
      ),
      width: 160,
    },
    {
      title: "Arguments",
      dataIndex: "args",
      key: "args",
      render: (v: string) => (
        <span className="font-mono text-xs text-gray-600 truncate block max-w-xs" title={v}>
          {v}
        </span>
      ),
    },
    {
      title: "Blocked Reason",
      dataIndex: "reason",
      key: "reason",
      render: (v: string) => <span className="text-sm text-gray-700">{v}</span>,
    },
    {
      title: "Severity",
      dataIndex: "severity",
      key: "severity",
      render: (v: string) => <SeverityTag s={v} />,
      width: 110,
    },
  ];

  return (
    <div className="p-6 max-w-[1100px] mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => router.push("/agent-monitoring")}
          className="flex-shrink-0"
        >
          Back
        </Button>
        <div>
          <h1 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <RobotOutlined className="text-indigo-500" />
            Claims Agent
            <Tag color="red" icon={<WarningOutlined />} className="ml-1">
              CRITICAL · 41 violations
            </Tag>
          </h1>
          <p className="text-sm text-gray-500">United Health · Drift: 87% · Last active: 2 min ago</p>
        </div>
      </div>

      {/* Root Cause Analysis */}
      <Card
        title={
          <span className="flex items-center gap-2 text-red-700">
            <BugOutlined /> Root Cause Analysis
          </span>
        }
        className="border-red-200"
        headStyle={{ borderBottom: "1px solid #fecaca", backgroundColor: "#fff7f7" }}
      >
        <div className="space-y-3">
          <Alert
            type="error"
            showIcon
            message="Agent returned executable code to the user in a compliance-answering context"
            description="The Claims Agent was deployed to answer member compliance questions, but it is calling bash_tool and code_execution — tools outside its permitted scope. This creates a data exfiltration risk and violates HIPAA access controls."
            className="rounded-lg"
          />

          <div className="grid grid-cols-3 gap-3 mt-2">
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-center">
              <div className="text-2xl font-bold text-red-600">23</div>
              <div className="text-xs text-red-500 mt-1">bash_tool calls (24h)</div>
            </div>
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-center">
              <div className="text-2xl font-bold text-red-600">18</div>
              <div className="text-xs text-red-500 mt-1">code_execution calls (24h)</div>
            </div>
            <div className="rounded-lg border border-orange-200 bg-orange-50 p-3 text-center">
              <div className="text-2xl font-bold text-orange-600">87%</div>
              <div className="text-xs text-orange-500 mt-1">Behavioral drift from baseline</div>
            </div>
          </div>

          <div className="mt-3">
            <div className="text-sm font-semibold text-gray-700 mb-2">Investigation timeline</div>
            <Timeline
              items={[
                {
                  color: "red",
                  children: (
                    <div>
                      <span className="font-semibold text-sm">09:31 — First code_execution detected</span>
                      <div className="text-xs text-gray-500">Agent wrote member claims data to local file</div>
                    </div>
                  ),
                },
                {
                  color: "red",
                  children: (
                    <div>
                      <span className="font-semibold text-sm">09:35 — bash_tool called with curl</span>
                      <div className="text-xs text-gray-500">Agent attempted to POST to internal-api/claims/dump</div>
                    </div>
                  ),
                },
                {
                  color: "orange",
                  children: (
                    <div>
                      <span className="font-semibold text-sm">09:41 — LiteLLM policy guardrail blocked both tools</span>
                      <div className="text-xs text-gray-500">Tool policy enforcement stopped the calls before execution</div>
                    </div>
                  ),
                },
                {
                  color: "blue",
                  dot: <ClockCircleOutlined />,
                  children: (
                    <div>
                      <span className="font-semibold text-sm text-blue-600">Now — Awaiting policy update</span>
                      <div className="text-xs text-gray-500">Block bash_tool and code_execution permanently for this agent</div>
                    </div>
                  ),
                },
              ]}
            />
          </div>
        </div>
      </Card>

      {/* Violations log */}
      <Card
        title={
          <span className="flex items-center gap-2">
            <ExclamationCircleOutlined className="text-red-500" />
            Violation Log — Last 24 Hours
            <Badge count={VIOLATIONS.length} color="red" className="ml-1" />
          </span>
        }
      >
        <Table
          dataSource={VIOLATIONS}
          columns={violationColumns}
          rowKey="id"
          size="small"
          pagination={false}
          rowClassName="hover:bg-red-50"
          scroll={{ x: true }}
        />
      </Card>

      {/* Policy fix */}
      <Card
        title={
          <span className="flex items-center gap-2 text-indigo-700">
            <LockOutlined /> Suggested Fix — Tool Policy
          </span>
        }
        headStyle={{ borderBottom: "1px solid #e0e7ff", backgroundColor: "#f5f3ff" }}
      >
        {policyApplied ? (
          <Alert
            type="success"
            showIcon
            icon={<CheckCircleOutlined />}
            message="Policy applied! bash_tool and code_execution are now blocked for Claims Agent."
            description="The agent has been re-tested. Tool calls return a policy-violation error before reaching the LLM. You can verify by re-running a sample request below."
            className="rounded-lg"
          />
        ) : (
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              The following tools were detected outside the Claims Agent's permitted scope. Select the tools to block and
              apply the policy to prevent future violations.
            </p>

            <div className="space-y-2">
              <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Currently Allowed (keep)</div>
              <div className="flex flex-wrap gap-2">
                {ALLOWED_TOOLS.map((t) => (
                  <Tag key={t} color="green" icon={<CheckCircleOutlined />} className="font-mono">
                    {t}
                  </Tag>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Block these tools</div>
              <div className="flex flex-wrap gap-3">
                {BLOCKED_TOOLS_SUGGESTED.map((t) => {
                  const selected = blockedTools.includes(t);
                  return (
                    <button
                      key={t}
                      onClick={() => toggleBlock(t)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 font-mono font-semibold text-sm transition-all ${
                        selected
                          ? "border-red-500 bg-red-50 text-red-700 shadow-sm"
                          : "border-gray-300 bg-white text-gray-600 hover:border-red-300"
                      }`}
                    >
                      <LockOutlined />
                      {t}
                      {selected && <CheckCircleOutlined className="text-red-500" />}
                    </button>
                  );
                })}
              </div>
              <p className="text-xs text-gray-400">Click to toggle — selected tools will be added to the deny list</p>
            </div>

            <div className="flex gap-3 pt-2">
              <Button
                type="primary"
                danger
                disabled={blockedTools.length === 0}
                onClick={applyPolicy}
                icon={<LockOutlined />}
                size="large"
              >
                Apply Policy ({blockedTools.length} tool{blockedTools.length !== 1 ? "s" : ""} blocked)
              </Button>
              <Button
                onClick={() => setBlockedTools(BLOCKED_TOOLS_SUGGESTED)}
                size="large"
              >
                Select All Suggested
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
