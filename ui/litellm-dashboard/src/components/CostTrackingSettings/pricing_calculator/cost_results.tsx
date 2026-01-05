import React from "react";
import { Text } from "@tremor/react";
import { Card, Statistic, Row, Col, Divider, Spin } from "antd";
import { DollarOutlined, LoadingOutlined } from "@ant-design/icons";
import { CostEstimateResponse } from "../types";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import ExportDropdown from "./export_dropdown";

interface CostResultsProps {
  result: CostEstimateResponse | null;
  loading: boolean;
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

const CostResults: React.FC<CostResultsProps> = ({ result, loading }) => {
  if (!result && !loading) {
    return (
      <div className="py-8 text-center border border-dashed border-gray-300 rounded-lg">
        <Text className="text-gray-500">
          Select a model to see cost estimates
        </Text>
      </div>
    );
  }

  if (loading && !result) {
    return (
      <div className="py-8 text-center">
        <Spin indicator={<LoadingOutlined spin />} />
        <Text className="text-gray-500 block mt-2">Calculating costs...</Text>
      </div>
    );
  }

  if (!result) return null;

  return (
    <div className="space-y-4">
      <Divider />

      <div className="mb-4 flex items-center justify-between">
        <div>
          <Text className="text-lg font-semibold text-gray-900">Cost Estimate</Text>
          <Text className="text-sm text-gray-500 block mt-1">
            Model: {result.model} {result.provider && `(${result.provider})`}
          </Text>
        </div>
        <div className="flex items-center gap-2">
          {loading && <Spin indicator={<LoadingOutlined spin />} size="small" />}
          <ExportDropdown result={result} />
        </div>
      </div>

      <Card size="small" title="Per-Request Cost Breakdown">
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Total Cost"
              value={formatCost(result.cost_per_request)}
              valueStyle={{ color: "#1890ff", fontSize: "18px" }}
              prefix={<DollarOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Input Cost"
              value={formatCost(result.input_cost_per_request)}
              valueStyle={{ fontSize: "16px" }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Output Cost"
              value={formatCost(result.output_cost_per_request)}
              valueStyle={{ fontSize: "16px" }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Margin/Fee"
              value={formatCost(result.margin_cost_per_request)}
              valueStyle={{
                fontSize: "16px",
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
                valueStyle={{ color: "#52c41a", fontSize: "18px" }}
                prefix={<DollarOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Input Cost"
                value={formatCost(result.daily_input_cost)}
                valueStyle={{ fontSize: "16px" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Output Cost"
                value={formatCost(result.daily_output_cost)}
                valueStyle={{ fontSize: "16px" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Margin/Fee"
                value={formatCost(result.daily_margin_cost)}
                valueStyle={{
                  fontSize: "16px",
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
                valueStyle={{ color: "#722ed1", fontSize: "18px" }}
                prefix={<DollarOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Input Cost"
                value={formatCost(result.monthly_input_cost)}
                valueStyle={{ fontSize: "16px" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Output Cost"
                value={formatCost(result.monthly_output_cost)}
                valueStyle={{ fontSize: "16px" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Margin/Fee"
                value={formatCost(result.monthly_margin_cost)}
                valueStyle={{
                  fontSize: "16px",
                  color: (result.monthly_margin_cost ?? 0) > 0 ? "#faad14" : undefined,
                }}
              />
            </Col>
          </Row>
        </Card>
      )}

      {(result.input_cost_per_token || result.output_cost_per_token) && (
        <div className="text-sm text-gray-500 mt-4">
          <Text className="font-medium">Token Pricing: </Text>
          {result.input_cost_per_token && (
            <span>Input: ${formatNumberWithCommas(result.input_cost_per_token * 1_000_000, 2)}/1M tokens</span>
          )}
          {result.input_cost_per_token && result.output_cost_per_token && " | "}
          {result.output_cost_per_token && (
            <span>Output: ${formatNumberWithCommas(result.output_cost_per_token * 1_000_000, 2)}/1M tokens</span>
          )}
        </div>
      )}
    </div>
  );
};

export default CostResults;
