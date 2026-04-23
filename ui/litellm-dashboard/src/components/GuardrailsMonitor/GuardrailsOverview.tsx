import {
  DownloadOutlined,
  RiseOutlined,
  SafetyOutlined,
  SettingOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Card, Col, Grid, Title } from "@tremor/react";
import { Button, Spin, Table } from "antd";
import type { ColumnsType } from "antd/es/table";
import React, { useMemo, useState } from "react";
import { getGuardrailsUsageOverview } from "@/components/networking";
import { type PerformanceRow } from "./mockData";
import { EvaluationSettingsModal } from "./EvaluationSettingsModal";
import { MetricCard } from "./MetricCard";
import { ScoreChart } from "./ScoreChart";

interface GuardrailsOverviewProps {
  accessToken?: string | null;
  startDate: string;
  endDate: string;
  onSelectGuardrail: (id: string) => void;
}

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

function computeMetricsFromRows(data: PerformanceRow[]) {
  const totalRequests = data.reduce((sum, r) => sum + r.requestsEvaluated, 0);
  const totalBlocked = data.reduce(
    (sum, r) => sum + Math.round((r.requestsEvaluated * r.failRate) / 100),
    0
  );
  const passRate =
    totalRequests > 0 ? ((1 - totalBlocked / totalRequests) * 100).toFixed(1) : "0";
  const withLat = data.filter((r) => r.avgLatency != null);
  const avgLatency =
    withLat.length > 0
      ? Math.round(withLat.reduce((sum, r) => sum + (r.avgLatency ?? 0), 0) / withLat.length)
      : 0;
  return { totalRequests, totalBlocked, passRate, avgLatency, count: data.length };
}

export function GuardrailsOverview({
  accessToken = null,
  startDate,
  endDate,
  onSelectGuardrail,
}: GuardrailsOverviewProps) {
  const [sortBy, setSortBy] = useState<SortKey>("failRate");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [evaluationModalOpen, setEvaluationModalOpen] = useState(false);

  const { data: guardrailsData, isLoading: guardrailsLoading, error: guardrailsError } = useQuery({
    queryKey: ["guardrails-usage-overview", startDate, endDate],
    queryFn: () => getGuardrailsUsageOverview(accessToken!, startDate, endDate),
    enabled: !!accessToken,
  });

  const activeData: PerformanceRow[] = guardrailsData?.rows ?? [];
  const metrics = useMemo(() => {
    if (guardrailsData) {
      return {
        totalRequests: guardrailsData.totalRequests ?? 0,
        totalBlocked: guardrailsData.totalBlocked ?? 0,
        passRate: String(guardrailsData.passRate ?? 0),
        avgLatency: activeData.length ? Math.round(activeData.reduce((s, r) => s + (r.avgLatency ?? 0), 0) / activeData.length) : 0,
        count: activeData.length,
      };
    }
    return computeMetricsFromRows(activeData);
  }, [guardrailsData, activeData]);
  const chartData = guardrailsData?.chart;
  const sorted = useMemo(() => {
    return [...activeData].sort((a, b) => {
      const mult = sortDir === "desc" ? -1 : 1;
      const aVal = a[sortBy] ?? 0;
      const bVal = b[sortBy] ?? 0;
      return (Number(aVal) - Number(bVal)) * mult;
    });
  }, [activeData, sortBy, sortDir]);
  const isLoading = guardrailsLoading;
  const error = guardrailsError;

  const columns: ColumnsType<PerformanceRow> = [
    {
      title: "Guardrail",
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
      render: (v?: number) => (
        <span
          className={
            v == null ? "text-gray-400" : v > 150 ? "text-red-600" : v > 50 ? "text-amber-600" : "text-green-600"
          }
        >
          {v != null ? `${v}ms` : "—"}
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

  const sortableKeys: SortKey[] = ["failRate", "requestsEvaluated", "avgLatency"];
  const handleTableChange = (_pagination: unknown, _filters: unknown, sorter: unknown) => {
    const s = sorter as { field?: keyof PerformanceRow; order?: string };
    if (s?.field && sortableKeys.includes(s.field as SortKey)) {
      setSortBy(s.field as SortKey);
      setSortDir(s.order === "ascend" ? "asc" : "desc");
    }
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
            Monitor guardrail performance across all requests
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button type="default" icon={<DownloadOutlined />} title="Coming soon">
            Export Data
          </Button>
        </div>
      </div>

      <Grid numItems={2} numItemsLg={5} className="gap-4 mb-6 items-stretch">
        <Col className="flex flex-col">
          <MetricCard label="Total Evaluations" value={metrics.totalRequests.toLocaleString()} />
        </Col>
        <Col className="flex flex-col">
          <MetricCard
            label="Blocked Requests"
            value={metrics.totalBlocked.toLocaleString()}
            valueColor="text-red-600"
            icon={<WarningOutlined className="text-red-400" />}
          />
        </Col>
        <Col className="flex flex-col">
          <MetricCard
            label="Pass Rate"
            value={`${metrics.passRate}%`}
            valueColor="text-green-600"
            icon={<RiseOutlined className="text-green-400" />}
          />
        </Col>
        <Col className="flex flex-col">
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
          />
        </Col>
        <Col className="flex flex-col">
          <MetricCard
            label="Active Guardrails"
            value={metrics.count}
          />
        </Col>
      </Grid>

      <div className="mb-6">
        <ScoreChart data={chartData} />
      </div>

      <Card className="bg-white border border-gray-200 rounded-lg">
        {(isLoading || error) && (
          <div className="px-6 py-4 border-b border-gray-200 flex items-center gap-2">
            {isLoading && <Spin size="small" />}
            {error && <span className="text-sm text-red-600">Failed to load data. Try again.</span>}
          </div>
        )}
        <div className="px-6 py-4 border-b border-gray-200 flex items-start justify-between gap-4">
          <div>
            <Title className="text-base font-semibold text-gray-900">
              Guardrail Performance
            </Title>
            <p className="text-xs text-gray-500 mt-0.5">
              Click a guardrail to view details, logs, and configuration
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="default"
              icon={<SettingOutlined />}
              onClick={() => setEvaluationModalOpen(true)}
              title="Evaluation settings"
            />
          </div>
        </div>
        <Table
          columns={columns}
          dataSource={sorted}
          rowKey="id"
          pagination={false}
          loading={isLoading}
          onChange={handleTableChange}
          locale={activeData.length === 0 && !isLoading ? { emptyText: "No data for this period" } : undefined}
          onRow={(row) => ({
            onClick: () => onSelectGuardrail(row.id),
            style: { cursor: "pointer" },
          })}
        />
      </Card>

      <EvaluationSettingsModal
        open={evaluationModalOpen}
        onClose={() => setEvaluationModalOpen(false)}
        accessToken={accessToken}
      />
    </div>
  );
}
