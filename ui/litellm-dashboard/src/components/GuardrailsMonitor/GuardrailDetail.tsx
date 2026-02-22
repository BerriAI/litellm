import {
  ArrowLeftOutlined,
  BellOutlined,
  PlayCircleOutlined,
  SafetyOutlined,
  SettingOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Card, Col, Grid, Title } from "@tremor/react";
import { Button, Spin, Tabs } from "antd";
import React, { useMemo, useState } from "react";
import {
  formatDate,
  getGuardrailsUsageDetail,
  getGuardrailsUsageLogs,
} from "@/components/networking";
import { EvaluationSettingsModal } from "./EvaluationSettingsModal";
import { LogViewer } from "./LogViewer";
import { MetricCard } from "./MetricCard";
import type { LogEntry } from "./mockData";

interface GuardrailDetailProps {
  guardrailId: string;
  onBack: () => void;
  accessToken?: string | null;
}

const statusColors: Record<
  string,
  { bg: string; text: string; dot: string }
> = {
  healthy: { bg: "bg-green-50", text: "text-green-700", dot: "bg-green-500" },
  warning: { bg: "bg-amber-50", text: "text-amber-700", dot: "bg-amber-500" },
  critical: { bg: "bg-red-50", text: "text-red-700", dot: "bg-red-500" },
};

const defaultEnd = new Date();
const defaultStart = new Date();
defaultStart.setDate(defaultStart.getDate() - 7);

export function GuardrailDetail({
  guardrailId,
  onBack,
  accessToken = null,
}: GuardrailDetailProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const [evaluationModalOpen, setEvaluationModalOpen] = useState(false);
  const [startDate] = useState(() => formatDate(defaultStart));
  const [endDate] = useState(() => formatDate(defaultEnd));
  const [logsPage, setLogsPage] = useState(1);
  const logsPageSize = 50;

  const { data: detailData, isLoading: detailLoading, error: detailError } = useQuery({
    queryKey: ["guardrails-usage-detail", guardrailId, startDate, endDate],
    queryFn: () => getGuardrailsUsageDetail(accessToken!, guardrailId, startDate, endDate),
    enabled: !!accessToken && !!guardrailId,
  });
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ["guardrails-usage-logs", guardrailId, logsPage, logsPageSize],
    queryFn: () =>
      getGuardrailsUsageLogs(accessToken!, {
        guardrailId,
        page: logsPage,
        pageSize: logsPageSize,
        startDate,
        endDate,
      }),
    enabled: !!accessToken && !!guardrailId,
  });

  const logs: LogEntry[] = useMemo(() => {
    const list = logsData?.logs ?? [];
    return list.map((l: Record<string, unknown>) => ({
      id: l.id as string,
      timestamp: l.timestamp as string,
      action: l.action as "blocked" | "passed" | "flagged",
      score: l.score as number | undefined,
      model: l.model as string | undefined,
      input_snippet: l.input_snippet as string | undefined,
      output_snippet: l.output_snippet as string | undefined,
      reason: l.reason as string | undefined,
    }));
  }, [logsData?.logs]);

  const data = detailData
    ? {
        name: detailData.guardrail_name,
        description: detailData.description ?? "",
        status: detailData.status,
        provider: detailData.provider,
        type: detailData.type,
        requestsEvaluated: detailData.requestsEvaluated,
        failRate: detailData.failRate,
        avgScore: detailData.avgScore,
        avgLatency: detailData.avgLatency,
      }
    : {
        name: guardrailId,
        description: "",
        status: "healthy",
        provider: "—",
        type: "—",
        requestsEvaluated: 0,
        failRate: 0,
        avgScore: undefined as number | undefined,
        avgLatency: undefined as number | undefined,
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
        <Button
          type="link"
          icon={<ArrowLeftOutlined />}
          onClick={onBack}
          className="pl-0 mb-4"
        >
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
            <Button type="default" icon={<PlayCircleOutlined />} title="Coming soon">
              Re-run AI
            </Button>
            <Button
              type="default"
              icon={<SettingOutlined />}
              onClick={() => setEvaluationModalOpen(true)}
              title="Evaluation settings"
            />
            <Button
              type="default"
              icon={<BellOutlined />}
              title="Coming soon"
              className="opacity-75"
            >
              Notify
            </Button>
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
          <Grid numItems={2} numItemsMd={5} className="gap-4">
            <Col>
              <MetricCard label="Requests Evaluated" value={data.requestsEvaluated.toLocaleString()} />
            </Col>
            <Col>
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
            <Col>
              <MetricCard
                label="Avg. latency added"
                value={
                  data.avgLatency != null ? `${Math.round(data.avgLatency)}ms` : "—"
                }
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
          </Grid>

          <Card className="bg-white border border-gray-200 rounded-lg p-6">
            <Title className="text-base font-semibold text-gray-900 mb-1">
              Root Cause Analysis
            </Title>
            <p className="text-xs text-gray-500 mb-4">Common patterns in failing requests</p>
            <div className="space-y-3">
              <div className="flex items-start gap-3 p-3 bg-red-50 rounded-lg border border-red-100">
                <WarningOutlined className="text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-red-800">
                    High sensitivity to medical terminology
                  </p>
                  <p className="text-xs text-red-600 mt-0.5">
                    34% of blocked requests contain common medical terms (e.g., &quot;symptoms&quot;,
                    &quot;treatment&quot;, &quot;medication&quot;) that are benign in context.
                    Consider adding an allowlist or relaxing sensitivity for these categories.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-amber-50 rounded-lg border border-amber-100">
                <WarningOutlined className="text-amber-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-amber-800">
                    False positives on educational content
                  </p>
                  <p className="text-xs text-amber-600 mt-0.5">
                    22% of blocked requests are educational queries about safety topics. The guardrail
                    is flagging the topic itself rather than harmful intent.
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <WarningOutlined className="text-gray-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-gray-800">
                    Sensitivity may be too aggressive
                  </p>
                  <p className="text-xs text-gray-600 mt-0.5">
                    Many blocked requests may be false positives. Consider relaxing sensitivity or
                    adding allowlisted patterns to reduce blocks by ~40% while maintaining safety.
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <LogViewer
            guardrailName={data.name}
            filterAction="blocked"
            logs={logs}
            logsLoading={logsLoading}
            totalLogs={logsData?.total ?? 0}
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
