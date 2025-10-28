import React from "react";
import { Tooltip } from "antd";
import {
  ClockCircleOutlined,
  NumberOutlined,
  ImportOutlined,
  ExportOutlined,
  BulbOutlined,
  ToolOutlined,
} from "@ant-design/icons";

export interface TokenUsage {
  completionTokens?: number;
  promptTokens?: number;
  totalTokens?: number;
  reasoningTokens?: number;
}

interface ResponseMetricsProps {
  timeToFirstToken?: number; // in milliseconds
  usage?: TokenUsage;
  toolName?: string;
}

const ResponseMetrics: React.FC<ResponseMetricsProps> = ({ timeToFirstToken, usage, toolName }) => {
  if (!timeToFirstToken && !usage) return null;

  return (
    <div className="response-metrics mt-2 pt-2 border-t border-gray-100 text-xs text-gray-500 flex flex-wrap gap-3">
      {timeToFirstToken !== undefined && (
        <Tooltip title="Time to first token">
          <div className="flex items-center">
            <ClockCircleOutlined className="mr-1" />
            <span>{(timeToFirstToken / 1000).toFixed(2)}s</span>
          </div>
        </Tooltip>
      )}

      {usage?.promptTokens !== undefined && (
        <Tooltip title="Prompt tokens">
          <div className="flex items-center">
            <ImportOutlined className="mr-1" />
            <span>In: {usage.promptTokens}</span>
          </div>
        </Tooltip>
      )}

      {usage?.completionTokens !== undefined && (
        <Tooltip title="Completion tokens">
          <div className="flex items-center">
            <ExportOutlined className="mr-1" />
            <span>Out: {usage.completionTokens}</span>
          </div>
        </Tooltip>
      )}

      {usage?.reasoningTokens !== undefined && (
        <Tooltip title="Reasoning tokens">
          <div className="flex items-center">
            <BulbOutlined className="mr-1" />
            <span>Reasoning: {usage.reasoningTokens}</span>
          </div>
        </Tooltip>
      )}

      {usage?.totalTokens !== undefined && (
        <Tooltip title="Total tokens">
          <div className="flex items-center">
            <NumberOutlined className="mr-1" />
            <span>Total: {usage.totalTokens}</span>
          </div>
        </Tooltip>
      )}

      {toolName && (
        <Tooltip title="Tool used">
          <div className="flex items-center">
            <ToolOutlined className="mr-1" />
            <span>Tool: {toolName}</span>
          </div>
        </Tooltip>
      )}
    </div>
  );
};

export default ResponseMetrics;
