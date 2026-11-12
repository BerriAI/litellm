"use client";

import React from "react";
import { useTranslation } from "react-i18next";

import { ErrorState } from "@/components/shared/ErrorState";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { NUMERIC_CELL_CLASS, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  formatCount,
  formatLatency,
  formatShare,
  formatUsd,
  formatUsdTotal,
  modelLabel,
  type TaskDetail,
} from "./taskInsightsData";

interface TaskDetailSheetProps {
  open: boolean;
  detail: TaskDetail | undefined;
  loading: boolean;
  error: string | null;
  sampleData: boolean;
  onRetry: () => void;
  onClose: () => void;
  onClearModel: () => void;
}

function Criteria({ label, text }: { label: string; text: string }) {
  if (!text) return null;
  return (
    <div>
      <dt className="text-xs font-medium text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 text-sm">{text}</dd>
    </div>
  );
}

function DetailBody({ detail, sampleData, onClearModel }: { detail: TaskDetail; sampleData: boolean; onClearModel: () => void }) {
  const { t } = useTranslation();
  const configs = new Map(detail.model_configs.map((config) => [config.model_config_id, config]));
  const nameOf = (id: string | null | undefined) => {
    if (!id) return t("usage.taskInsights.detail.unknownDeployment");
    const config = configs.get(id);
    return config ? modelLabel(config) : id;
  };
  const definition = detail.definition;
  return (
    <div className="flex flex-col gap-5 overflow-y-auto px-4 pb-6">
      <div className="flex flex-wrap gap-2">
        <Badge variant="secondary">
          {t("usage.taskInsights.detail.requests", { requests: formatCount(detail.summary.request_count) })}
        </Badge>
        <Badge variant="secondary">{formatShare(detail.summary.share)}</Badge>
        <Badge variant="secondary">
          {t("usage.taskInsights.detail.spend", {
            spend: formatUsdTotal(detail.summary.spend) ?? t("usage.taskInsights.noData"),
          })}
        </Badge>
        {sampleData && <Badge variant="outline">{t("usage.taskInsights.sampleBadge")}</Badge>}
      </div>
      {definition && (
        <dl className="flex flex-col gap-3">
          <Criteria label={t("usage.taskInsights.detail.includes")} text={definition.inclusion_criteria} />
          <Criteria label={t("usage.taskInsights.detail.excludes")} text={definition.exclusion_criteria} />
          {definition.required_capabilities.length > 0 && (
            <Criteria
              label={t("usage.taskInsights.detail.requires")}
              text={definition.required_capabilities.join(", ")}
            />
          )}
          {definition.examples.length > 0 && (
            <div>
              <dt className="text-xs font-medium text-muted-foreground">{t("usage.taskInsights.detail.examples")}</dt>
              <dd>
                <ul className="mt-1 list-disc space-y-1 pl-5 text-sm">
                  {definition.examples.map((example) => (
                    <li key={example}>{example}</li>
                  ))}
                </ul>
              </dd>
            </div>
          )}
        </dl>
      )}
      <section className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">{t("usage.taskInsights.detail.byModel")}</h3>
        {detail.entries.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("usage.taskInsights.detail.noModelEvidence")}</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("usage.taskInsights.detail.model")}</TableHead>
                <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.taskInsights.breakdown.requests")}</TableHead>
                <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.taskInsights.heatmap.metrics.cost")}</TableHead>
                <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.taskInsights.heatmap.metrics.latency")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {detail.entries.map((entry) => (
                <TableRow
                  key={entry.model_config_id}
                  data-state={detail.model_config_id === entry.model_config_id ? "selected" : undefined}
                >
                  <TableCell>{nameOf(entry.model_config_id)}</TableCell>
                  <TableCell className={NUMERIC_CELL_CLASS}>{formatCount(entry.request_count)}</TableCell>
                  <TableCell className={NUMERIC_CELL_CLASS}>
                    {formatUsd(entry.avg_cost_usd) ?? t("usage.taskInsights.noData")}
                    <span className="block text-xs text-muted-foreground">
                      {t("usage.taskInsights.heatmap.observations", { observations: entry.cost_observations })}
                    </span>
                  </TableCell>
                  <TableCell className={NUMERIC_CELL_CLASS}>
                    {formatLatency(entry.p95_latency_ms) ?? t("usage.taskInsights.noData")}
                    <span className="block text-xs text-muted-foreground">
                      {t("usage.taskInsights.heatmap.observations", { observations: entry.latency_observations })}
                    </span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </section>
      <section className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-medium">
            {detail.model_config_id
              ? t("usage.taskInsights.detail.recentOnModel", { model: nameOf(detail.model_config_id) })
              : t("usage.taskInsights.detail.recent")}
          </h3>
          {detail.model_config_id && (
            <Button variant="ghost" size="sm" onClick={onClearModel}>
              {t("usage.taskInsights.detail.allModels")}
            </Button>
          )}
        </div>
        {detail.requests.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("usage.taskInsights.detail.noRequests")}</p>
        ) : (
          <ul className="flex flex-col divide-y rounded-lg border">
            {detail.requests.map((request) => (
              <li key={request.request_id} className="flex flex-col gap-1 p-3">
                <p className="line-clamp-3 text-sm">
                  {request.excerpt ?? <span className="text-muted-foreground">{t("usage.taskInsights.detail.noText")}</span>}
                </p>
                <p className="flex flex-wrap gap-x-3 text-xs text-muted-foreground">
                  <span>{new Date(request.start_time).toLocaleString()}</span>
                  <span>{nameOf(request.model_config_id)}</span>
                  <span>{formatUsd(request.spend) ?? t("usage.taskInsights.detail.costUnknown")}</span>
                  <span>{formatLatency(request.latency_ms) ?? t("usage.taskInsights.detail.latencyUnknown")}</span>
                  {request.top_probability !== null && request.top_probability !== undefined && (
                    <span>
                      {t("usage.taskInsights.detail.probability", { probability: formatShare(request.top_probability) })}
                    </span>
                  )}
                  {request.delivery_count > 1 && (
                    <span>{t("usage.taskInsights.detail.deliveries", { deliveries: request.delivery_count })}</span>
                  )}
                  {request.session_id && (
                    <span className="truncate">{t("usage.taskInsights.detail.session", { session: request.session_id })}</span>
                  )}
                </p>
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-muted-foreground">{t("usage.taskInsights.detail.probabilityNote")}</p>
      </section>
    </div>
  );
}

export default function TaskDetailSheet({
  open,
  detail,
  loading,
  error,
  sampleData,
  onRetry,
  onClose,
  onClearModel,
}: TaskDetailSheetProps) {
  const { t } = useTranslation();
  return (
    <Sheet open={open} onOpenChange={(next) => !next && onClose()}>
      <SheetContent className="gap-3 data-[side=right]:w-full data-[side=right]:sm:max-w-[560px]">
        <SheetHeader>
          <SheetTitle>{detail?.summary.name ?? t("usage.taskInsights.detail.loadingTitle")}</SheetTitle>
          <SheetDescription>{detail?.summary.description ?? ""}</SheetDescription>
        </SheetHeader>
        {error ? (
          <div className="px-4">
            <ErrorState title={t("usage.taskInsights.detail.loadFailed")} message={error} onRetry={onRetry} />
          </div>
        ) : loading || !detail ? (
          <div role="status" aria-label={t("usage.taskInsights.loading")} className="flex flex-col gap-3 px-4">
            <Skeleton className="h-6 w-2/3" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        ) : (
          <DetailBody detail={detail} sampleData={sampleData} onClearModel={onClearModel} />
        )}
      </SheetContent>
    </Sheet>
  );
}
