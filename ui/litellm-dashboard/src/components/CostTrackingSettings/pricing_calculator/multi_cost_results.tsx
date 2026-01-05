import React, { useState } from "react";
import { Text, Button } from "@tremor/react";
import { Card, Statistic, Row, Col, Divider, Spin, Table, Tag } from "antd";
import { DollarOutlined, LoadingOutlined, DownOutlined, RightOutlined } from "@ant-design/icons";
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
    <div className="space-y-3">
      {loading && (
        <div className="flex items-center gap-2 text-gray-500 text-sm">
          <Spin indicator={<LoadingOutlined spin />} size="small" />
          <span>Updating...</span>
        </div>
      )}

      <Card size="small" title="Per-Request Cost Breakdown">
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Total Cost"
              value={formatCost(result.cost_per_request)}
              valueStyle={{ color: "#1890ff", fontSize: "16px" }}
              prefix={<DollarOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Input Cost"
              value={formatCost(result.input_cost_per_request)}
              valueStyle={{ fontSize: "14px" }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Output Cost"
              value={formatCost(result.output_cost_per_request)}
              valueStyle={{ fontSize: "14px" }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Margin/Fee"
              value={formatCost(result.margin_cost_per_request)}
              valueStyle={{
                fontSize: "14px",
                color: result.margin_cost_per_request > 0 ? "#faad14" : undefined,
              }}
            />
          </Col>
        </Row>
      </Card>

      {result.daily_cost !== null && (
        <Card
          size="small"
          title={`Daily Costs (${formatRequests(result.num_requests_per_day)} requests/day)`}
        >
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="Total Daily"
                value={formatCost(result.daily_cost)}
                valueStyle={{ color: "#52c41a", fontSize: "16px" }}
                prefix={<DollarOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Input Cost"
                value={formatCost(result.daily_input_cost)}
                valueStyle={{ fontSize: "14px" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Output Cost"
                value={formatCost(result.daily_output_cost)}
                valueStyle={{ fontSize: "14px" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Margin/Fee"
                value={formatCost(result.daily_margin_cost)}
                valueStyle={{
                  fontSize: "14px",
                  color: (result.daily_margin_cost ?? 0) > 0 ? "#faad14" : undefined,
                }}
              />
            </Col>
          </Row>
        </Card>
      )}

      {result.monthly_cost !== null && (
        <Card
          size="small"
          title={`Monthly Costs (${formatRequests(result.num_requests_per_month)} requests/month)`}
        >
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="Total Monthly"
                value={formatCost(result.monthly_cost)}
                valueStyle={{ color: "#722ed1", fontSize: "16px" }}
                prefix={<DollarOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Input Cost"
                value={formatCost(result.monthly_input_cost)}
                valueStyle={{ fontSize: "14px" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Output Cost"
                value={formatCost(result.monthly_output_cost)}
                valueStyle={{ fontSize: "14px" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Margin/Fee"
                value={formatCost(result.monthly_margin_cost)}
                valueStyle={{
                  fontSize: "14px",
                  color: (result.monthly_margin_cost ?? 0) > 0 ? "#faad14" : undefined,
                }}
              />
            </Col>
          </Row>
        </Card>
      )}

      {(result.input_cost_per_token || result.output_cost_per_token) && (
        <div className="text-xs text-gray-500">
          <span className="font-medium">Token Pricing: </span>
          {result.input_cost_per_token && (
            <span>Input: ${formatNumberWithCommas(result.input_cost_per_token * 1_000_000, 2)}/1M</span>
          )}
          {result.input_cost_per_token && result.output_cost_per_token && " | "}
          {result.output_cost_per_token && (
            <span>Output: ${formatNumberWithCommas(result.output_cost_per_token * 1_000_000, 2)}/1M</span>
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
      <div className="py-8 text-center border border-dashed border-gray-300 rounded-lg">
        <Text className="text-gray-500">
          Select models above to see cost estimates
        </Text>
      </div>
    );
  }

  if (!hasAnyResult && isAnyLoading) {
    return (
      <div className="py-8 text-center">
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

  const summaryColumns = [
    {
      title: "Model",
      dataIndex: "model",
      key: "model",
      render: (text: string, record: { id: string; provider?: string | null }) => (
        <div>
          <span className="font-medium">{text}</span>
          {record.provider && (
            <Tag color="blue" className="ml-2 text-xs">
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
      render: (value: number) => formatCost(value),
    },
    {
      title: "Daily",
      dataIndex: "daily_cost",
      key: "daily_cost",
      render: (value: number | null) => formatCost(value),
    },
    {
      title: "Monthly",
      dataIndex: "monthly_cost",
      key: "monthly_cost",
      render: (value: number | null) => formatCost(value),
    },
    {
      title: "",
      key: "expand",
      width: 50,
      render: (_: unknown, record: { id: string }) => (
        <Button
          size="xs"
          variant="light"
          onClick={() => toggleExpanded(record.id)}
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
    daily_cost: e.result!.daily_cost,
    monthly_cost: e.result!.monthly_cost,
  }));

  return (
    <div className="space-y-4">
      <Divider />

      <div className="flex items-center justify-between">
        <div>
          <Text className="text-lg font-semibold text-gray-900">Cost Estimates</Text>
          <Text className="text-sm text-gray-500 block mt-1">
            {validEntries.length} model{validEntries.length !== 1 ? "s" : ""} configured
          </Text>
        </div>
        <div className="flex items-center gap-2">
          {isAnyLoading && <Spin indicator={<LoadingOutlined spin />} size="small" />}
          <MultiExportDropdown multiResult={multiResult} />
        </div>
      </div>

      {/* Totals Summary */}
      {validEntries.length > 1 && (
        <Card size="small" className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
          <div className="flex items-center justify-between">
            <Text className="font-semibold text-gray-800">Combined Totals</Text>
          </div>
          <Row gutter={16} className="mt-3">
            <Col span={8}>
              <Statistic
                title="Total Per Request"
                value={formatCost(multiResult.totals.cost_per_request)}
                valueStyle={{ color: "#1890ff", fontSize: "20px" }}
                prefix={<DollarOutlined />}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Total Daily"
                value={formatCost(multiResult.totals.daily_cost)}
                valueStyle={{ color: "#52c41a", fontSize: "20px" }}
                prefix={<DollarOutlined />}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Total Monthly"
                value={formatCost(multiResult.totals.monthly_cost)}
                valueStyle={{ color: "#722ed1", fontSize: "20px" }}
                prefix={<DollarOutlined />}
              />
            </Col>
          </Row>
        </Card>
      )}

      {/* Per-Model Summary Table */}
      <Table
        columns={summaryColumns}
        dataSource={summaryData}
        pagination={false}
        size="small"
        expandable={{
          expandedRowKeys: Array.from(expandedModels),
          expandedRowRender: (record) => {
            const entry = validEntries.find((e) => e.entry.id === record.id);
            if (!entry?.result) return null;
            return (
              <div className="py-4 px-2">
                <SingleModelBreakdown result={entry.result} loading={entry.loading} />
              </div>
            );
          },
          showExpandColumn: false,
        }}
      />

      {/* Error Messages */}
      {multiResult.entries
        .filter((e) => e.error)
        .map((e) => (
          <div key={e.entry.id} className="text-sm text-red-500 bg-red-50 p-2 rounded">
            <span className="font-medium">{e.entry.model || "Unknown model"}: </span>
            {e.error}
          </div>
        ))}
    </div>
  );
};

export default MultiCostResults;

