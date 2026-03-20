import React, { useState } from "react";
import { Text, Button } from "@tremor/react";
import { Card, Statistic, Row, Col, Divider, Spin, Table, Tag } from "antd";
import { LoadingOutlined, DownOutlined, RightOutlined } from "@ant-design/icons";
import { CostEstimateResponse } from "../types";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { MultiModelResult } from "./types";
import MultiExportDropdown from "./multi_export_dropdown";

interface MultiCostResultsProps {
  multiResult: MultiModelResult;
  timePeriod: "day" | "month";
}

const formatCost = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "-";
  if (value === 0) return "$0";
  if (value < 0.0001) return `$${value.toExponential(2)}`;
  if (value < 1) return `$${value.toFixed(4)}`;
  return `$${formatNumberWithCommas(value, 2, true)}`;
};

const formatRequests = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "-";
  return formatNumberWithCommas(value, 0, true);
};

const SingleModelBreakdown: React.FC<{
  result: CostEstimateResponse;
  loading: boolean;
  timePeriod: "day" | "month";
}> = ({ result, loading, timePeriod }) => {
  const periodLabel = timePeriod === "day" ? "Daily" : "Monthly";
  const periodCost = timePeriod === "day" ? result.daily_cost : result.monthly_cost;
  const periodInputCost = timePeriod === "day" ? result.daily_input_cost : result.monthly_input_cost;
  const periodOutputCost = timePeriod === "day" ? result.daily_output_cost : result.monthly_output_cost;
  const periodMarginCost = timePeriod === "day" ? result.daily_margin_cost : result.monthly_margin_cost;
  const periodRequests = timePeriod === "day" ? result.num_requests_per_day : result.num_requests_per_month;

  return (
    <div className="space-y-3 bg-gray-50 p-4 rounded-lg">
      {loading && (
        <div className="flex items-center gap-2 text-gray-500 text-sm">
          <Spin indicator={<LoadingOutlined spin />} size="small" />
          <span>Updating...</span>
        </div>
      )}

      <div className="grid grid-cols-4 gap-4">
        <div>
          <Text className="text-xs text-gray-500 block">Total/Request</Text>
          <Text className="text-base font-semibold text-blue-600">{formatCost(result.cost_per_request)}</Text>
        </div>
        <div>
          <Text className="text-xs text-gray-500 block">Input Cost</Text>
          <Text className="text-sm">{formatCost(result.input_cost_per_request)}</Text>
        </div>
        <div>
          <Text className="text-xs text-gray-500 block">Output Cost</Text>
          <Text className="text-sm">{formatCost(result.output_cost_per_request)}</Text>
        </div>
        <div>
          <Text className="text-xs text-gray-500 block">Margin Fee</Text>
          <Text className={`text-sm ${result.margin_cost_per_request > 0 ? "text-amber-600" : ""}`}>
            {formatCost(result.margin_cost_per_request)}
          </Text>
        </div>
      </div>

      {periodCost !== null && (
        <div className="grid grid-cols-4 gap-4 pt-2 border-t border-gray-200">
          <div>
            <Text className="text-xs text-gray-500 block">{periodLabel} Total ({formatRequests(periodRequests)} req)</Text>
            <Text className={`text-base font-semibold ${timePeriod === "day" ? "text-green-600" : "text-purple-600"}`}>
              {formatCost(periodCost)}
            </Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">{periodLabel} Input</Text>
            <Text className="text-sm">{formatCost(periodInputCost)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">{periodLabel} Output</Text>
            <Text className="text-sm">{formatCost(periodOutputCost)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">{periodLabel} Margin Fee</Text>
            <Text className={`text-sm ${(periodMarginCost ?? 0) > 0 ? "text-amber-600" : ""}`}>
              {formatCost(periodMarginCost)}
            </Text>
          </div>
        </div>
      )}

      {(result.input_cost_per_token || result.output_cost_per_token) && (
        <div className="text-xs text-gray-400 pt-2 border-t border-gray-200">
          Token Pricing: {" "}
          {result.input_cost_per_token && (
            <span>Input ${formatNumberWithCommas(result.input_cost_per_token * 1_000_000, 2)}/1M</span>
          )}
          {result.input_cost_per_token && result.output_cost_per_token && " | "}
          {result.output_cost_per_token && (
            <span>Output ${formatNumberWithCommas(result.output_cost_per_token * 1_000_000, 2)}/1M</span>
          )}
        </div>
      )}
    </div>
  );
};

const MultiCostResults: React.FC<MultiCostResultsProps> = ({ multiResult, timePeriod }) => {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  const validEntries = multiResult.entries.filter((e) => e.result !== null);
  const loadingEntries = multiResult.entries.filter((e) => e.loading);
  const errorEntries = multiResult.entries.filter((e) => e.error !== null);
  const hasAnyResult = validEntries.length > 0;
  const isAnyLoading = loadingEntries.length > 0;
  const hasAnyError = errorEntries.length > 0;

  // Show empty state only if no results, not loading, and no errors
  if (!hasAnyResult && !isAnyLoading && !hasAnyError) {
    return (
      <div className="py-6 text-center border border-dashed border-gray-300 rounded-lg bg-gray-50">
        <Text className="text-gray-500">
          Select models above to see cost estimates
        </Text>
      </div>
    );
  }

  // Show loading state only if loading and no results/errors yet
  if (!hasAnyResult && isAnyLoading && !hasAnyError) {
    return (
      <div className="py-6 text-center">
        <Spin indicator={<LoadingOutlined spin />} />
        <Text className="text-gray-500 block mt-2">Calculating costs...</Text>
      </div>
    );
  }

  // Show errors-only view when there are errors but no valid results
  if (!hasAnyResult && hasAnyError) {
    return (
      <div className="space-y-4">
        <Divider className="my-4" />
        <div className="flex items-center justify-between">
          <Text className="text-base font-semibold text-gray-900">Cost Estimates</Text>
          {isAnyLoading && <Spin indicator={<LoadingOutlined spin />} size="small" />}
        </div>
        {/* Error Messages */}
        {errorEntries.map((e) => (
          <div key={e.entry.id} className="text-sm text-red-600 bg-red-50 p-3 rounded-lg border border-red-200">
            <span className="font-medium">{e.entry.model || "Unknown model"}: </span>
            {e.error}
          </div>
        ))}
      </div>
    );
  }

  const toggleExpanded = (id: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const hasMargin = multiResult.totals.margin_per_request > 0;

  const periodLabel = timePeriod === "day" ? "Daily" : "Monthly";
  const periodCostKey = timePeriod === "day" ? "daily_cost" : "monthly_cost";

  const summaryColumns = [
    {
      title: "Model",
      dataIndex: "model",
      key: "model",
      render: (text: string, record: { id: string; provider?: string | null; error?: string | null; loading?: boolean; hasZeroCost?: boolean | null }) => (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm">{text}</span>
            {record.provider && (
              <Tag color="blue" className="text-xs">
                {record.provider}
              </Tag>
            )}
            {record.loading && (
              <Spin indicator={<LoadingOutlined spin />} size="small" />
            )}
          </div>
          {record.error && (
            <div className="text-xs text-red-600 bg-red-50 px-2 py-1 rounded">
              ⚠️ {record.error}
            </div>
          )}
          {record.hasZeroCost && !record.error && (
            <div className="text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded">
              ⚠️ No pricing data found for this model. Set base_model in config.
            </div>
          )}
        </div>
      ),
    },
    {
      title: "Per Request",
      dataIndex: "cost_per_request",
      key: "cost_per_request",
      align: "right" as const,
      render: (value: number | null, record: { error?: string | null }) => (
        record.error ? <span className="text-gray-400">-</span> : <span className="font-mono text-sm">{formatCost(value)}</span>
      ),
    },
    {
      title: "Margin Fee",
      dataIndex: "margin_cost_per_request",
      key: "margin_cost_per_request",
      align: "right" as const,
      render: (value: number | null, record: { error?: string | null }) => (
        record.error ? <span className="text-gray-400">-</span> : (
          <span className={`font-mono text-sm ${(value ?? 0) > 0 ? "text-amber-600" : "text-gray-400"}`}>
            {formatCost(value)}
          </span>
        )
      ),
    },
    {
      title: periodLabel,
      dataIndex: periodCostKey,
      key: "period_cost",
      align: "right" as const,
      render: (value: number | null, record: { error?: string | null }) => (
        record.error ? <span className="text-gray-400">-</span> : <span className="font-mono text-sm">{formatCost(value)}</span>
      ),
    },
    {
      title: "",
      key: "expand",
      width: 40,
      render: (_: unknown, record: { id: string; error?: string | null }) => (
        record.error ? null : (
          <Button
            size="xs"
            variant="light"
            onClick={() => toggleExpanded(record.id)}
            className="text-gray-400 hover:text-gray-600"
          >
            {expandedModels.has(record.id) ? <DownOutlined /> : <RightOutlined />}
          </Button>
        )
      ),
    },
  ];

  // Include both valid results and errors in the table data
  const allEntriesWithModels = multiResult.entries.filter((e) => e.entry.model);
  const summaryData = allEntriesWithModels.map((e) => ({
    key: e.entry.id,
    id: e.entry.id,
    model: e.result?.model || e.entry.model,
    provider: e.result?.provider,
    cost_per_request: e.result?.cost_per_request ?? null,
    margin_cost_per_request: e.result?.margin_cost_per_request ?? null,
    daily_cost: e.result?.daily_cost ?? null,
    monthly_cost: e.result?.monthly_cost ?? null,
    error: e.error,
    loading: e.loading,
    hasZeroCost: e.result && e.result.cost_per_request === 0,
  }));

  return (
    <div className="space-y-4">
      <Divider className="my-4" />

      <div className="flex items-center justify-between">
        <Text className="text-base font-semibold text-gray-900">Cost Estimates</Text>
        <div className="flex items-center gap-2">
          {isAnyLoading && <Spin indicator={<LoadingOutlined spin />} size="small" />}
          <MultiExportDropdown multiResult={multiResult} />
        </div>
      </div>

      {/* Combined Totals - Always show when there are results */}
      <Card size="small" className="bg-gradient-to-r from-slate-50 to-blue-50 border-slate-200">
        <Row gutter={[16, 8]}>
          <Col xs={24} sm={12}>
            <Statistic
              title={<span className="text-xs">Total Per Request</span>}
              value={formatCost(multiResult.totals.cost_per_request)}
              valueStyle={{ color: "#1890ff", fontSize: "18px", fontFamily: "monospace" }}
            />
          </Col>
          <Col xs={24} sm={12}>
            <Statistic
              title={<span className="text-xs">Total {periodLabel}</span>}
              value={formatCost(timePeriod === "day" ? multiResult.totals.daily_cost : multiResult.totals.monthly_cost)}
              valueStyle={{ color: timePeriod === "day" ? "#52c41a" : "#722ed1", fontSize: "18px", fontFamily: "monospace" }}
            />
          </Col>
        </Row>
        {hasMargin && (
          <Row gutter={[16, 8]} className="mt-3 pt-3 border-t border-slate-200">
            <Col xs={24} sm={12}>
              <div className="text-xs text-gray-500">Margin Fee/Request</div>
              <div className="text-sm font-mono text-amber-600">{formatCost(multiResult.totals.margin_per_request)}</div>
            </Col>
            <Col xs={24} sm={12}>
              <div className="text-xs text-gray-500">{periodLabel} Margin Fee</div>
              <div className="text-sm font-mono text-amber-600">
                {formatCost(timePeriod === "day" ? multiResult.totals.daily_margin : multiResult.totals.monthly_margin)}
              </div>
            </Col>
          </Row>
        )}
      </Card>

      {/* Per-Model Table */}
      {summaryData.length > 0 && (
        <Table
          columns={summaryColumns}
          dataSource={summaryData}
          pagination={false}
          size="small"
          className="border border-gray-200 rounded-lg"
          expandable={{
            expandedRowKeys: Array.from(expandedModels),
            expandedRowRender: (record) => {
              const entry = validEntries.find((e) => e.entry.id === record.id);
              if (!entry?.result) return null;
              return (
                <div className="py-2">
                  <SingleModelBreakdown result={entry.result} loading={entry.loading} timePeriod={timePeriod} />
                </div>
              );
            },
            showExpandColumn: false,
          }}
        />
      )}
    </div>
  );
};

export default MultiCostResults;
