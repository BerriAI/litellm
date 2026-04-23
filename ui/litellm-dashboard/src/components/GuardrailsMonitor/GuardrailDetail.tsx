import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Settings, Shield, AlertTriangle } from "lucide-react";
import React, { useMemo, useState } from "react";
import {
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
  startDate: string;
  endDate: string;
}

const statusStyles: Record<
  string,
  { bg: string; text: string; dot: string }
> = {
  healthy: {
    bg: "bg-emerald-50 dark:bg-emerald-950/30",
    text: "text-emerald-700 dark:text-emerald-400",
    dot: "bg-emerald-500",
  },
  warning: {
    bg: "bg-amber-50 dark:bg-amber-950/30",
    text: "text-amber-700 dark:text-amber-400",
    dot: "bg-amber-500",
  },
  critical: {
    bg: "bg-red-50 dark:bg-red-950/30",
    text: "text-red-700 dark:text-red-400",
    dot: "bg-red-500",
  },
};

export function GuardrailDetail({
  guardrailId,
  onBack,
  accessToken = null,
  startDate,
  endDate,
}: GuardrailDetailProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const [evaluationModalOpen, setEvaluationModalOpen] = useState(false);
  const [logsPage] = useState(1);
  const logsPageSize = 50;

  const {
    data: detailData,
    isLoading: detailLoading,
    error: detailError,
  } = useQuery({
    queryKey: ["guardrails-usage-detail", guardrailId, startDate, endDate],
    queryFn: () =>
      getGuardrailsUsageDetail(accessToken!, guardrailId, startDate, endDate),
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
  const statusStyle = statusStyles[data.status] ?? statusStyles.healthy;

  if (detailLoading && !detailData) {
    return (
      <div className="flex items-center justify-center py-12">
        <Skeleton className="h-10 w-10 rounded-full" />
      </div>
    );
  }
  if (detailError && !detailData) {
    return (
      <div>
        <Button variant="link" onClick={onBack} className="pl-0 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Overview
        </Button>
        <p className="text-destructive">Failed to load guardrail details.</p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <Button variant="link" onClick={onBack} className="pl-0 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Overview
        </Button>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <Shield className="h-5 w-5 text-muted-foreground" />
              <h1 className="text-xl font-semibold text-foreground">
                {data.name}
              </h1>
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium rounded-full ${statusStyle.bg} ${statusStyle.text}`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${statusStyle.dot}`}
                />
                {data.status.charAt(0).toUpperCase() + data.status.slice(1)}
              </span>
            </div>
            <p className="text-sm text-muted-foreground ml-8">
              {data.description}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200 dark:bg-indigo-950/30 dark:text-indigo-300 dark:border-indigo-800">
              {data.provider}
            </span>
            <Button
              variant="outline"
              size="icon"
              onClick={() => setEvaluationModalOpen(true)}
              title="Evaluation settings"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
        </TabsList>
      </Tabs>

      {activeTab === "overview" && (
        <div className="space-y-6 mt-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <MetricCard
              label="Requests Evaluated"
              value={data.requestsEvaluated.toLocaleString()}
            />
            <MetricCard
              label="Fail Rate"
              value={`${data.failRate}%`}
              valueColor={
                data.failRate > 15
                  ? "text-red-600 dark:text-red-400"
                  : data.failRate > 5
                    ? "text-amber-600 dark:text-amber-400"
                    : "text-emerald-600 dark:text-emerald-400"
              }
              subtitle={`${Math.round((data.requestsEvaluated * data.failRate) / 100).toLocaleString()} blocked`}
              icon={
                data.failRate > 15 ? (
                  <AlertTriangle className="h-4 w-4 text-red-400" />
                ) : undefined
              }
            />
            <MetricCard
              label="Avg. latency added"
              value={
                data.avgLatency != null
                  ? `${Math.round(data.avgLatency)}ms`
                  : "—"
              }
              valueColor={
                data.avgLatency != null
                  ? data.avgLatency > 150
                    ? "text-red-600 dark:text-red-400"
                    : data.avgLatency > 50
                      ? "text-amber-600 dark:text-amber-400"
                      : "text-emerald-600 dark:text-emerald-400"
                  : "text-muted-foreground"
              }
              subtitle={data.avgLatency != null ? "Per request (avg)" : "No data"}
            />
          </div>

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
