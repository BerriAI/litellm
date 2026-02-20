import {
  CheckCircleOutlined,
  DownloadOutlined,
  FileTextOutlined,
  PlayCircleOutlined,
  RiseOutlined,
  SafetyOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { Card, Col, Grid, Title } from "@tremor/react";
import { Button, Spin, Table } from "antd";
import type { ColumnsType } from "antd/es/table";
import React, { useEffect, useMemo, useState } from "react";
import {
  guardrailsTable,
  policiesTable,
  type PerformanceRow,
} from "./mockData";
import { MetricCard } from "./MetricCard";
import { ScoreChart } from "./ScoreChart";

interface GuardrailsOverviewProps {
  onSelectGuardrail: (id: string) => void;
}

type ViewMode = "guardrails" | "policies";
type SortKey =
  | "failRate"
  | "requestsEvaluated"
  | "avgLatency"
  | "falsePositiveRate"
  | "falseNegativeRate";

const providerColors: Record<string, string> = {
  Bedrock: "bg-orange-100 text-orange-700 border-orange-200",
  "Google Cloud": "bg-sky-100 text-sky-700 border-sky-200",
  LiteLLM: "bg-indigo-100 text-indigo-700 border-indigo-200",
  Custom: "bg-gray-100 text-gray-600 border-gray-200",
};

function computeMetrics(data: PerformanceRow[]) {
  const totalRequests = data.reduce((sum, r) => sum + r.requestsEvaluated, 0);
  const totalBlocked = data.reduce(
    (sum, r) => sum + Math.round((r.requestsEvaluated * r.failRate) / 100),
    0
  );
  const passRate =
    totalRequests > 0 ? ((1 - totalBlocked / totalRequests) * 100).toFixed(1) : "0";
  const avgLatency =
    data.length > 0
      ? Math.round(data.reduce((sum, r) => sum + r.avgLatency, 0) / data.length)
      : 0;
  const p95Latency =
    data.length > 0
      ? Math.round(data.reduce((sum, r) => sum + r.p95Latency, 0) / data.length)
      : 0;
  return { totalRequests, totalBlocked, passRate, avgLatency, p95Latency, count: data.length };
}

type RerunState = "idle" | "running" | "done";

export function GuardrailsOverview({ onSelectGuardrail }: GuardrailsOverviewProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("guardrails");
  const [sortBy, setSortBy] = useState<SortKey>("failRate");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [rerunState, setRerunState] = useState<RerunState>("idle");

  useEffect(() => {
    if (rerunState !== "done") return;
    const t = setTimeout(() => setRerunState("idle"), 4000);
    return () => clearTimeout(t);
  }, [rerunState]);

  const activeData = viewMode === "guardrails" ? guardrailsTable : policiesTable;
  const metrics = useMemo(() => computeMetrics(activeData), [activeData]);
  const sorted = useMemo(() => {
    return [...activeData].sort((a, b) => {
      const mult = sortDir === "desc" ? -1 : 1;
      return (a[sortBy] - b[sortBy]) * mult;
    });
  }, [activeData, sortBy, sortDir]);

  const isGuardrails = viewMode === "guardrails";

  const columns: ColumnsType<PerformanceRow> = [
    {
      title: isGuardrails ? "Guardrail" : "Policy",
      dataIndex: "name",
      key: "name",
      render: (name: string, row) => (
        <button
          type="button"
          className="text-sm font-medium text-gray-900 hover:text-indigo-600 text-left"
          onClick={() => onSelectGuardrail(row.id)}
        >
          {name}
        </button>
      ),
    },
    {
      title: "Provider",
      dataIndex: "provider",
      key: "provider",
      render: (provider: string) => (
        <span
          className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded border ${
            providerColors[provider] ?? providerColors.Custom
          }`}
        >
          {provider}
        </span>
      ),
    },
    {
      title: "Requests",
      dataIndex: "requestsEvaluated",
      key: "requestsEvaluated",
      align: "right",
      sorter: true,
      sortOrder: sortBy === "requestsEvaluated" ? (sortDir === "desc" ? "descend" : "ascend") : null,
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: "Fail Rate",
      dataIndex: "failRate",
      key: "failRate",
      align: "right",
      sorter: true,
      sortOrder: sortBy === "failRate" ? (sortDir === "desc" ? "descend" : "ascend") : null,
      render: (v: number, row) => (
        <span
          className={
            v > 15 ? "text-red-600" : v > 5 ? "text-amber-600" : "text-green-600"
          }
        >
          {v}%
          {row.trend === "up" && <span className="ml-1 text-xs text-red-400">↑</span>}
          {row.trend === "down" && <span className="ml-1 text-xs text-green-400">↓</span>}
        </span>
      ),
    },
    {
      title: "Avg. latency added",
      dataIndex: "avgLatency",
      key: "avgLatency",
      align: "right",
      sorter: true,
      sortOrder: sortBy === "avgLatency" ? (sortDir === "desc" ? "descend" : "ascend") : null,
      render: (v: number, row: PerformanceRow) => (
        <span>
          <span
            className={
              v > 150 ? "text-red-600" : v > 50 ? "text-amber-600" : "text-green-600"
            }
          >
            {v}ms
          </span>
          <span className="block text-xs text-gray-500">p95: {row.p95Latency}ms</span>
        </span>
      ),
    },
    {
      title: "False Pos %",
      dataIndex: "falsePositiveRate",
      key: "falsePositiveRate",
      align: "right",
      sorter: true,
      sortOrder:
        sortBy === "falsePositiveRate"
          ? sortDir === "desc"
            ? "descend"
            : "ascend"
          : null,
      render: (v: number) => (
        <span
          className={
            v > 20 ? "text-red-600" : v > 10 ? "text-amber-600" : "text-green-600"
          }
        >
          {v}%
        </span>
      ),
    },
    {
      title: "False Neg %",
      dataIndex: "falseNegativeRate",
      key: "falseNegativeRate",
      align: "right",
      sorter: true,
      sortOrder:
        sortBy === "falseNegativeRate"
          ? sortDir === "desc"
            ? "descend"
            : "ascend"
          : null,
      render: (v: number) => (
        <span
          className={
            v > 5 ? "text-red-600" : v > 2 ? "text-amber-600" : "text-green-600"
          }
        >
          {v}%
        </span>
      ),
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      align: "center",
      render: (status: string) => (
        <span className="inline-flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${
              status === "healthy"
                ? "bg-green-500"
                : status === "warning"
                  ? "bg-amber-500"
                  : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-600 capitalize">{status}</span>
        </span>
      ),
    },
  ];

  const sortableKeys: SortKey[] = [
    "failRate",
    "requestsEvaluated",
    "avgLatency",
    "falsePositiveRate",
    "falseNegativeRate",
  ];
  const handleTableChange = (_pagination: unknown, _filters: unknown, sorter: unknown) => {
    const s = sorter as { field?: keyof PerformanceRow; order?: string };
    if (s?.field && sortableKeys.includes(s.field as SortKey)) {
      setSortBy(s.field as SortKey);
      setSortDir(s.order === "ascend" ? "asc" : "desc");
    }
  };

  const handleRerun = () => {
    if (rerunState !== "idle") return;
    setRerunState("running");
    setTimeout(() => setRerunState("done"), 2500);
  };

  return (
    <div>
      <div className="flex items-start justify-between mb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <SafetyOutlined className="text-lg text-indigo-500" />
            <h1 className="text-xl font-semibold text-gray-900">Guardrails Monitor</h1>
          </div>
          <p className="text-sm text-gray-500">
            {isGuardrails
              ? "Monitor guardrail performance across all requests"
              : "Monitor policy enforcement across all requests"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-600 bg-white border border-gray-200 rounded-md px-3 py-2">
            12 Feb, 12:07 – 19 Feb, 12:07
          </span>
          <Button type="primary" icon={<DownloadOutlined />}>
            Export Data
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-1 p-1 bg-gray-100 rounded-lg w-fit mb-6">
        <button
          type="button"
          onClick={() => {
            setViewMode("guardrails");
            setSortBy("failRate");
            setSortDir("desc");
          }}
          className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 text-sm font-medium rounded-md transition-colors ${
            isGuardrails ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"
          }`}
        >
          <SafetyOutlined /> Guardrail Performance
        </button>
        <button
          type="button"
          onClick={() => {
            setViewMode("policies");
            setSortBy("failRate");
            setSortDir("desc");
          }}
          className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 text-sm font-medium rounded-md transition-colors ${
            !isGuardrails ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"
          }`}
        >
          <FileTextOutlined /> Policy Performance
        </button>
      </div>

      <Grid numItems={2} numItemsLg={5} className="gap-4 mb-6">
        <Col>
          <MetricCard label="Total Requests Evaluated" value={metrics.totalRequests.toLocaleString()} />
        </Col>
        <Col>
          <MetricCard
            label="Blocked Requests"
            value={metrics.totalBlocked.toLocaleString()}
            valueColor="text-red-600"
            icon={<WarningOutlined className="text-red-400" />}
          />
        </Col>
        <Col>
          <MetricCard
            label="Pass Rate"
            value={`${metrics.passRate}%`}
            valueColor="text-green-600"
            icon={<RiseOutlined className="text-green-400" />}
          />
        </Col>
        <Col>
          <MetricCard
            label="Avg. latency added"
            value={`${metrics.avgLatency}ms`}
            valueColor={
              metrics.avgLatency > 150
                ? "text-red-600"
                : metrics.avgLatency > 50
                  ? "text-amber-600"
                  : "text-green-600"
            }
            subtitle={`p95: ${metrics.p95Latency}ms`}
          />
        </Col>
        <Col>
          <MetricCard
            label={isGuardrails ? "Active Guardrails" : "Active Policies"}
            value={metrics.count}
          />
        </Col>
      </Grid>

      <div className="mb-6">
        <ScoreChart />
      </div>

      <Card className="bg-white border border-gray-200 rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200 flex items-start justify-between gap-4">
          <div>
            <Title className="text-base font-semibold text-gray-900">
              {isGuardrails ? "Guardrail Performance" : "Policy Performance"}
            </Title>
            <p className="text-xs text-gray-500 mt-0.5">
              {isGuardrails
                ? "Click a guardrail to view details, logs, and configuration"
                : "Click a policy to view details, logs, and configuration"}
            </p>
          </div>
          <Button
            type="default"
            icon={
              rerunState === "idle" ? (
                <PlayCircleOutlined />
              ) : rerunState === "done" ? (
                <CheckCircleOutlined className="text-green-600" />
              ) : (
                <Spin size="small" />
              )
            }
            disabled={rerunState === "running"}
            onClick={handleRerun}
          >
            {rerunState === "idle"
              ? "Re-run AI on last 100 logs"
              : rerunState === "running"
                ? "Re-running on 100 logs…"
                : "Re-run complete"}
          </Button>
        </div>
        <Table
          columns={columns}
          dataSource={sorted}
          rowKey="id"
          pagination={false}
          onChange={handleTableChange}
          onRow={(row) => ({
            onClick: () => onSelectGuardrail(row.id),
            style: { cursor: "pointer" },
          })}
        />
      </Card>
    </div>
  );
}
