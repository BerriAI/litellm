import React from "react";
import { Tooltip } from "antd";
import {
  ClockCircleOutlined,
  NumberOutlined,
  ImportOutlined,
  ExportOutlined,
  BulbOutlined,
  ToolOutlined,
  DollarOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";

export interface TokenUsage {
  completionTokens?: number;
  promptTokens?: number;
  totalTokens?: number;
  reasoningTokens?: number;
  cost?: number;
}

interface ResponseMetricsProps {
  timeToFirstToken?: number;
  totalLatency?: number;
  usage?: TokenUsage;
  toolName?: string;
}

const ResponseMetrics: React.FC<ResponseMetricsProps> = ({ timeToFirstToken, totalLatency, usage, toolName }) => {
  const { t } = useTranslation();
  if (!timeToFirstToken && !totalLatency && !usage) return null;

  return (
    <div className="response-metrics mt-2 pt-2 border-t border-gray-100 text-xs text-gray-500 flex flex-wrap gap-3">
      {timeToFirstToken !== undefined && (
        <Tooltip title={t("playground.responseMetrics.timeToFirstToken")}>
          <div className="flex items-center">
            <ClockCircleOutlined className="mr-1" />
            <span>{t("playground.responseMetrics.ttft", { value: (timeToFirstToken / 1000).toFixed(2) })}</span>
          </div>
        </Tooltip>
      )}

      {totalLatency !== undefined && (
        <Tooltip title={t("playground.responseMetrics.totalLatency")}>
          <div className="flex items-center">
            <ClockCircleOutlined className="mr-1" />
            <span>
              {t("playground.responseMetrics.totalLatencyValue", { value: (totalLatency / 1000).toFixed(2) })}
            </span>
          </div>
        </Tooltip>
      )}

      {usage?.promptTokens !== undefined && (
        <Tooltip title={t("playground.responseMetrics.promptTokens")}>
          <div className="flex items-center">
            <ImportOutlined className="mr-1" />
            <span>{t("playground.responseMetrics.in", { count: usage.promptTokens })}</span>
          </div>
        </Tooltip>
      )}

      {usage?.completionTokens !== undefined && (
        <Tooltip title={t("playground.responseMetrics.completionTokens")}>
          <div className="flex items-center">
            <ExportOutlined className="mr-1" />
            <span>{t("playground.responseMetrics.out", { count: usage.completionTokens })}</span>
          </div>
        </Tooltip>
      )}

      {usage?.reasoningTokens !== undefined && (
        <Tooltip title={t("playground.responseMetrics.reasoningTokens")}>
          <div className="flex items-center">
            <BulbOutlined className="mr-1" />
            <span>{t("playground.responseMetrics.reasoning", { count: usage.reasoningTokens })}</span>
          </div>
        </Tooltip>
      )}

      {usage?.totalTokens !== undefined && (
        <Tooltip title={t("playground.responseMetrics.totalTokens")}>
          <div className="flex items-center">
            <NumberOutlined className="mr-1" />
            <span>{t("playground.responseMetrics.total", { count: usage.totalTokens })}</span>
          </div>
        </Tooltip>
      )}

      {usage?.cost !== undefined && (
        <Tooltip title={t("playground.responseMetrics.cost")}>
          <div className="flex items-center">
            <DollarOutlined className="mr-1" />
            <span>${usage.cost.toFixed(6)}</span>
          </div>
        </Tooltip>
      )}

      {toolName && (
        <Tooltip title={t("playground.responseMetrics.toolUsed")}>
          <div className="flex items-center">
            <ToolOutlined className="mr-1" />
            <span>{t("playground.responseMetrics.toolName", { name: toolName })}</span>
          </div>
        </Tooltip>
      )}
    </div>
  );
};

export default ResponseMetrics;
