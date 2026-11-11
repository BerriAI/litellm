import React from "react";
import { useTranslation } from "react-i18next";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  Clock,
  Database,
  DatabaseBackup,
  DollarSign,
  Hash,
  History,
  Lightbulb,
  Wrench,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { PROMPT_CACHE_CREATION_TOOLTIP, PROMPT_CACHE_READ_TOOLTIP } from "@/utils/promptCacheUsage";

export interface TokenUsage {
  completionTokens?: number;
  promptTokens?: number;
  totalTokens?: number;
  reasoningTokens?: number;
  cacheReadTokens?: number;
  cacheCreationTokens?: number;
  cost?: number;
  servedFromResponseCache?: boolean;
}

interface ResponseMetricsProps {
  timeToFirstToken?: number;
  totalLatency?: number;
  usage?: TokenUsage;
  toolName?: string;
}

interface MetricItemProps {
  label: string;
  tooltip: string;
  icon: React.ReactNode;
  value: string;
}

function MetricItem({ label, tooltip, icon, value }: MetricItemProps) {
  return (
    <Tooltip>
      <TooltipTrigger render={<div className="flex items-center gap-1" aria-label={`${label}: ${value}`} />}>
        {icon}
        <span>
          {label}: {value}
        </span>
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  );
}

function ResponseCacheIndicator() {
  const { t } = useTranslation();
  return (
    <MetricItem
      label={t("chatUi.responseMetrics.labels.responseCache")}
      tooltip={t("chatUi.responseMetrics.tooltips.responseCache")}
      icon={<History className="size-3" aria-hidden="true" />}
      value={t("chatUi.responseMetrics.labels.responseCacheHit")}
    />
  );
}

function PromptCacheChips({ usage }: { usage?: TokenUsage }) {
  const { t } = useTranslation();
  if (usage?.servedFromResponseCache) {
    return <ResponseCacheIndicator />;
  }

  const readTokens = usage?.cacheReadTokens ?? 0;
  const creationTokens = usage?.cacheCreationTokens ?? 0;

  return (
    <>
      {readTokens > 0 && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.cacheRead")}
          tooltip={t(PROMPT_CACHE_READ_TOOLTIP)}
          icon={<Database className="size-3" aria-hidden="true" />}
          value={String(readTokens)}
        />
      )}

      {creationTokens > 0 && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.cacheWrite")}
          tooltip={t(PROMPT_CACHE_CREATION_TOOLTIP)}
          icon={<DatabaseBackup className="size-3" aria-hidden="true" />}
          value={String(creationTokens)}
        />
      )}
    </>
  );
}

const ResponseMetrics: React.FC<ResponseMetricsProps> = ({ timeToFirstToken, totalLatency, usage, toolName }) => {
  const { t } = useTranslation();
  if (!timeToFirstToken && !totalLatency && !usage) return null;

  return (
    <div className="response-metrics mt-2 flex flex-wrap gap-3 border-t border-border pt-2 text-xs text-muted-foreground">
      {timeToFirstToken !== undefined && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.timeToFirstToken")}
          tooltip={t("chatUi.responseMetrics.tooltips.timeToFirstToken")}
          icon={<Clock className="size-3" aria-hidden="true" />}
          value={`${(timeToFirstToken / 1000).toFixed(2)}s`}
        />
      )}

      {totalLatency !== undefined && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.totalLatency")}
          tooltip={t("chatUi.responseMetrics.tooltips.totalLatency")}
          icon={<Clock className="size-3" aria-hidden="true" />}
          value={`${(totalLatency / 1000).toFixed(2)}s`}
        />
      )}

      {usage?.promptTokens !== undefined && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.in")}
          tooltip={t("chatUi.responseMetrics.tooltips.promptTokens")}
          icon={<ArrowDownToLine className="size-3" aria-hidden="true" />}
          value={String(usage.promptTokens)}
        />
      )}

      <PromptCacheChips usage={usage} />

      {usage?.completionTokens !== undefined && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.out")}
          tooltip={t("chatUi.responseMetrics.tooltips.completionTokens")}
          icon={<ArrowUpFromLine className="size-3" aria-hidden="true" />}
          value={String(usage.completionTokens)}
        />
      )}

      {usage?.reasoningTokens !== undefined && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.reasoning")}
          tooltip={t("chatUi.responseMetrics.tooltips.reasoningTokens")}
          icon={<Lightbulb className="size-3" aria-hidden="true" />}
          value={String(usage.reasoningTokens)}
        />
      )}

      {usage?.totalTokens !== undefined && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.total")}
          tooltip={t("chatUi.responseMetrics.tooltips.totalTokens")}
          icon={<Hash className="size-3" aria-hidden="true" />}
          value={String(usage.totalTokens)}
        />
      )}

      {typeof usage?.cost === "number" && Number.isFinite(usage.cost) && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.cost")}
          tooltip={t("chatUi.responseMetrics.tooltips.cost")}
          icon={<DollarSign className="size-3" aria-hidden="true" />}
          value={`$${usage.cost.toFixed(6)}`}
        />
      )}

      {toolName && (
        <MetricItem
          label={t("chatUi.responseMetrics.labels.tool")}
          tooltip={t("chatUi.responseMetrics.tooltips.toolUsed")}
          icon={<Wrench className="size-3" aria-hidden="true" />}
          value={toolName}
        />
      )}
    </div>
  );
};

export default ResponseMetrics;
