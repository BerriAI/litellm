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
}> = ({ result, loading }) => {
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
          <Text className="text-xs text-gray-500 block">Margin/Fee</Text>
          <Text className={`text-sm ${result.margin_cost_per_request > 0 ? "text-amber-600" : ""}`}>
            {formatCost(result.margin_cost_per_request)}
          </Text>
        </div>
      </div>

      {result.daily_cost !== null && (
        <div className="grid grid-cols-4 gap-4 pt-2 border-t border-gray-200">
          <div>
            <Text className="text-xs text-gray-500 block">Daily Total ({formatRequests(result.num_requests_per_day)} req)</Text>
            <Text className="text-base font-semibold text-green-600">{formatCost(result.daily_cost)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">Daily Input</Text>
            <Text className="text-sm">{formatCost(result.daily_input_cost)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">Daily Output</Text>
            <Text className="text-sm">{formatCost(result.daily_output_cost)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">Daily Margin</Text>
            <Text className={`text-sm ${(result.daily_margin_cost ?? 0) > 0 ? "text-amber-600" : ""}`}>
              {formatCost(result.daily_margin_cost)}
            </Text>
          </div>
        </div>
      )}

      {result.monthly_cost !== null && (
        <div className="grid grid-cols-4 gap-4 pt-2 border-t border-gray-200">
          <div>
            <Text className="text-xs text-gray-500 block">Monthly Total ({formatRequests(result.num_requests_per_month)} req)</Text>
            <Text className="text-base font-semibold text-purple-600">{formatCost(result.monthly_cost)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">Monthly Input</Text>
            <Text className="text-sm">{formatCost(result.monthly_input_cost)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">Monthly Output</Text>
            <Text className="text-sm">{formatCost(result.monthly_output_cost)}</Text>
          </div>
          <div>
            <Text className="text-xs text-gray-500 block">Monthly Margin</Text>
            <Text className={`text-sm ${(result.monthly_margin_cost ?? 0) > 0 ? "text-amber-600" : ""}`}>
              {formatCost(result.monthly_margin_cost)}
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

const MultiCostResults: React.FC<MultiCostResultsProps> = ({ multiResult }) => {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  const validEntries = multiResult.entries.filter((e) => e.result !== null);
  const loadingEntries = multiResult.entries.filter((e) => e.loading);
  const hasAnyResult = validEntries.length > 0;
  const isAnyLoading = loadingEntries.length > 0;

  if (!hasAnyResult && !isAnyLoading) {
    return (
      <div className="py-6 text-center border border-dashed border-gray-300 rounded-lg bg-gray-50">
        <Text className="text-gray-500">
          Select models above to see cost estimates
        </Text>
      </div>
    );
  }

  if (!hasAnyResult && isAnyLoading) {
    return (
      <div className="py-6 text-center">
        <Spin indicator={<LoadingOutlined spin />} />
        <Text className="text-gray-500 block mt-2">Calculating costs...</Text>
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

  const summaryColumns = [
    {
      title: "Model",
      dataIndex: "model",
      key: "model",
      render: (text: string, record: { id: string; provider?: string | null }) => (
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{text}</span>
          {record.provider && (
            <Tag color="blue" className="text-xs">
              {record.provider}
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: "Per Request",
      dataIndex: "cost_per_request",
      key: "cost_per_request",
      align: "right" as const,
      render: (value: number) => <span className="font-mono text-sm">{formatCost(value)}</span>,
    },
    {
      title: "Margin",
      dataIndex: "margin_cost_per_request",
      key: "margin_cost_per_request",
      align: "right" as const,
      render: (value: number) => (
        <span className={`font-mono text-sm ${value > 0 ? "text-amber-600" : "text-gray-400"}`}>
          {formatCost(value)}
        </span>
      ),
    },
    {
      title: "Daily",
      dataIndex: "daily_cost",
      key: "daily_cost",
      align: "right" as const,
      render: (value: number | null) => <span className="font-mono text-sm">{formatCost(value)}</span>,
    },
    {
      title: "Monthly",
      dataIndex: "monthly_cost",
      key: "monthly_cost",
      align: "right" as const,
      render: (value: number | null) => <span className="font-mono text-sm">{formatCost(value)}</span>,
    },
    {
      title: "",
      key: "expand",
      width: 40,
      render: (_: unknown, record: { id: string }) => (
        <Button
          size="xs"
          variant="light"
          onClick={() => toggleExpanded(record.id)}
          className="text-gray-400 hover:text-gray-600"
        >
          {expandedModels.has(record.id) ? <DownOutlined /> : <RightOutlined />}
        </Button>
      ),
    },
  ];

  const summaryData = validEntries.map((e) => ({
    key: e.entry.id,
    id: e.entry.id,
    model: e.result!.model,
    provider: e.result!.provider,
    cost_per_request: e.result!.cost_per_request,
    margin_cost_per_request: e.result!.margin_cost_per_request,
    daily_cost: e.result!.daily_cost,
    monthly_cost: e.result!.monthly_cost,
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
          <Col xs={24} sm={8}>
            <Statistic
              title={<span className="text-xs">Total Per Request</span>}
              value={formatCost(multiResult.totals.cost_per_request)}
              valueStyle={{ color: "#1890ff", fontSize: "18px", fontFamily: "monospace" }}
            />
          </Col>
          <Col xs={24} sm={8}>
            <Statistic
              title={<span className="text-xs">Total Daily</span>}
              value={formatCost(multiResult.totals.daily_cost)}
              valueStyle={{ color: "#52c41a", fontSize: "18px", fontFamily: "monospace" }}
            />
          </Col>
          <Col xs={24} sm={8}>
            <Statistic
              title={<span className="text-xs">Total Monthly</span>}
              value={formatCost(multiResult.totals.monthly_cost)}
              valueStyle={{ color: "#722ed1", fontSize: "18px", fontFamily: "monospace" }}
            />
          </Col>
        </Row>
        {hasMargin && (
          <Row gutter={[16, 8]} className="mt-3 pt-3 border-t border-slate-200">
            <Col xs={24} sm={8}>
              <div className="text-xs text-gray-500">Margin/Request</div>
              <div className="text-sm font-mono text-amber-600">{formatCost(multiResult.totals.margin_per_request)}</div>
            </Col>
            <Col xs={24} sm={8}>
              <div className="text-xs text-gray-500">Daily Margin</div>
              <div className="text-sm font-mono text-amber-600">{formatCost(multiResult.totals.daily_margin)}</div>
            </Col>
            <Col xs={24} sm={8}>
              <div className="text-xs text-gray-500">Monthly Margin</div>
              <div className="text-sm font-mono text-amber-600">{formatCost(multiResult.totals.monthly_margin)}</div>
            </Col>
          </Row>
        )}
      </Card>

      {/* Per-Model Table */}
      {validEntries.length > 0 && (
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
                  <SingleModelBreakdown result={entry.result} loading={entry.loading} />
                </div>
              );
            },
            showExpandColumn: false,
          }}
        />
      )}

      {/* Error Messages */}
      {multiResult.entries
        .filter((e) => e.error)
        .map((e) => (
          <div key={e.entry.id} className="text-sm text-red-600 bg-red-50 p-3 rounded-lg border border-red-200">
            <span className="font-medium">{e.entry.model || "Unknown model"}: </span>
            {e.error}
          </div>
        ))}
    </div>
  );
};

export default MultiCostResults;
