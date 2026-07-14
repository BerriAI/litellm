import { ArrowLeftOutlined, SafetyOutlined, SettingOutlined, WarningOutlined } from "@ant-design/icons";
import { Button, Col, Row, Spin, Tabs } from "antd";
import React, { useState } from "react";
import { EvaluationSettingsModal } from "./EvaluationSettingsModal";
import { LogViewer } from "@/components/GuardrailsMonitor/LogViewer";
import { MetricCard } from "@/components/GuardrailsMonitor/MetricCard";
import { useGuardrailUsageDetail } from "@/app/(dashboard)/hooks/guardrailsMonitor/useGuardrailUsageDetail";
import { useGuardrailUsageLogs } from "@/app/(dashboard)/hooks/guardrailsMonitor/useGuardrailUsageLogs";

interface GuardrailDetailProps {
  guardrailId: string;
  onBack: () => void;
  accessToken?: string | null;
  startDate: string;
  endDate: string;
}

const statusColors: Record<string, { bg: string; text: string; dot: string }> = {
  healthy: { bg: "bg-green-50", text: "text-green-700", dot: "bg-green-500" },
  warning: { bg: "bg-amber-50", text: "text-amber-700", dot: "bg-amber-500" },
  critical: { bg: "bg-red-50", text: "text-red-700", dot: "bg-red-500" },
};

const LOGS_PAGE = 1;
const LOGS_PAGE_SIZE = 50;

export function GuardrailDetail({ guardrailId, onBack, accessToken = null, startDate, endDate }: GuardrailDetailProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const [evaluationModalOpen, setEvaluationModalOpen] = useState(false);

  const {
    data: detailData,
    isLoading: detailLoading,
    error: detailError,
  } = useGuardrailUsageDetail(guardrailId, startDate, endDate);
  const logsParams = {
    guardrailId,
    page: LOGS_PAGE,
    pageSize: LOGS_PAGE_SIZE,
    startDate,
    endDate,
  };
  const { data: logsData, isLoading: logsLoading } = useGuardrailUsageLogs(logsParams);

  const logs = logsData?.logs ?? [];
  const data = {
    name: detailData?.guardrail_name ?? guardrailId,
    description: detailData?.description ?? "",
    status: detailData?.status ?? "healthy",
    provider: detailData?.provider ?? "—",
    requestsEvaluated: detailData?.requestsEvaluated ?? 0,
    failRate: detailData?.failRate ?? 0,
    avgLatency: detailData?.avgLatency ?? null,
  };
  const statusStyle = statusColors[data.status] ?? statusColors.healthy;

  if (detailLoading && !detailData) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spin size="large" />
      </div>
    );
  }
  if (detailError && !detailData) {
    return (
      <div>
        <Button type="link" icon={<ArrowLeftOutlined />} onClick={onBack} className="pl-0 mb-4">
          Back to Overview
        </Button>
        <p className="text-red-600">Failed to load guardrail details.</p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <Button type="link" icon={<ArrowLeftOutlined />} onClick={onBack} className="pl-0 mb-4">
          Back to Overview
        </Button>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <SafetyOutlined className="text-xl text-gray-400" />
              <h1 className="text-xl font-semibold text-gray-900">{data.name}</h1>
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium rounded-full ${statusStyle.bg} ${statusStyle.text}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${statusStyle.dot}`} />
                {data.status.charAt(0).toUpperCase() + data.status.slice(1)}
              </span>
            </div>
            <p className="text-sm text-gray-500 ml-8">{data.description}</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200">
              {data.provider}
            </span>
            <Button
              type="default"
              icon={<SettingOutlined />}
              onClick={() => setEvaluationModalOpen(true)}
              title="Evaluation settings"
            />
          </div>
        </div>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          { key: "overview", label: "Overview" },
          { key: "logs", label: "Logs" },
        ]}
      />

      {activeTab === "overview" && (
        <div className="space-y-6 mt-4">
          <Row gutter={[16, 16]}>
            <Col xs={12} md={8}>
              <MetricCard label="Requests Evaluated" value={data.requestsEvaluated.toLocaleString()} />
            </Col>
            <Col xs={12} md={8}>
              <MetricCard
                label="Fail Rate"
                value={`${data.failRate}%`}
                valueColor={
                  data.failRate > 15 ? "text-red-600" : data.failRate > 5 ? "text-amber-600" : "text-green-600"
                }
                subtitle={`${Math.round((data.requestsEvaluated * data.failRate) / 100).toLocaleString()} blocked`}
                icon={data.failRate > 15 ? <WarningOutlined className="text-red-400" /> : undefined}
              />
            </Col>
            <Col xs={12} md={8}>
              <MetricCard
                label="Avg. latency added"
                value={data.avgLatency != null ? `${Math.round(data.avgLatency)}ms` : "—"}
                valueColor={
                  data.avgLatency != null
                    ? data.avgLatency > 150
                      ? "text-red-600"
                      : data.avgLatency > 50
                        ? "text-amber-600"
                        : "text-green-600"
                    : "text-gray-500"
                }
                subtitle={data.avgLatency != null ? "Per request (avg)" : "No data"}
              />
            </Col>
          </Row>

          <LogViewer
            guardrailName={data.name}
            filterAction="all"
            logs={logs}
            logsLoading={logsLoading}
            totalLogs={logsData?.total ?? 0}
            accessToken={accessToken}
            startDate={startDate}
            endDate={endDate}
          />
        </div>
      )}

      {activeTab === "logs" && (
        <div className="mt-4">
          <LogViewer
            guardrailName={data.name}
            logs={logs}
            logsLoading={logsLoading}
            totalLogs={logsData?.total ?? 0}
            accessToken={accessToken}
            startDate={startDate}
            endDate={endDate}
          />
        </div>
      )}

      <EvaluationSettingsModal
        open={evaluationModalOpen}
        onClose={() => setEvaluationModalOpen(false)}
        guardrailName={data.name}
        accessToken={accessToken}
      />
    </div>
  );
}
