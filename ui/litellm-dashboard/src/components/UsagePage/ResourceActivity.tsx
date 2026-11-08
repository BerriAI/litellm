"use client";

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useActivityDateRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";
import { useAggregatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/useAggregatedDailyActivity";
import { ModelSection } from "@/components/activity_metrics";
import { dailyActivityAggregatedCall } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { ChartLoader } from "@/components/shared/chart_loader";
import { ErrorState } from "@/components/shared/ErrorState";
import { hasProxyWideSpendView, spendScopeUserId } from "@/utils/roles";
import { EMPTY_DAILY_ACTIVITY_METADATA, toDailyData, type DailyActivityRequest } from "./dailyActivityApi";
import { keyDetailFromResponse, overallUsageMetrics, reportingKeyId } from "./keyActivityData";

interface ResourceActivityProps {
  scope: "key" | "team" | "user" | "model";
  id: string;
}

export default function ResourceActivity({ scope, id }: ResourceActivityProps) {
  const { t } = useTranslation();
  const { accessToken, userId, userRole } = useAuthorized();
  const { dateValue, onDateChange } = useActivityDateRange();
  const entity = scope === "team" ? "team" : "user";
  const entityId = scope === "team" || scope === "user" ? id : spendScopeUserId(userRole, userId);
  const apiKey = scope === "key" ? reportingKeyId(id) : undefined;
  const canView = scope !== "user" || hasProxyWideSpendView(userRole) || id === userId;
  const hasAccess = accessToken && id && canView;
  const scopedToViewer = (scope === "key" || scope === "model") && !hasProxyWideSpendView(userRole);
  const request: DailyActivityRequest | null =
    hasAccess && dateValue.from && dateValue.to
      ? {
          accessToken,
          startTime: dateValue.from,
          endTime: dateValue.to,
          entityIds: entityId ? [entityId] : null,
          apiKey,
          apiKeyLimit: 1,
          model: scope === "model" ? id : undefined,
          includeCurrentUtcDay: true,
        }
      : null;
  const { data, loading, failed, errorMessage, reload } = useAggregatedDailyActivity({
    fetch: () => dailyActivityAggregatedCall(entity, request as DailyActivityRequest),
    enabled: request !== null,
    deps: [accessToken, scope, id, entityId, dateValue.from, dateValue.to],
  });
  const metrics = useMemo(
    () =>
      (apiKey ? keyDetailFromResponse(data, apiKey, []) : undefined) ??
      overallUsageMetrics(toDailyData(data), data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA),
    [data, apiKey],
  );

  if (!request) return null;

  const content = (() => {
    if (loading) return <ChartLoader />;
    if (failed) return <ErrorState message={errorMessage} onRetry={reload} />;
    if (data.results.length === 0) {
      return <p className="py-8 text-center text-muted-foreground">{t("usage.keyActivity.noDailyDetails")}</p>;
    }
    return <ModelSection modelName={id} metrics={metrics} />;
  })();

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-end gap-4">
        <span className="text-sm text-muted-foreground">{t("shared.scopedSavingsTab.bucketedByUtc")}</span>
        <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
      </div>
      {scopedToViewer && <p className="text-sm text-muted-foreground">{t("shared.resourceActivity.ownUsage")}</p>}
      {scope === "model" && (
        <p className="text-sm text-muted-foreground">{t("shared.resourceActivity.modelScope", { model: id })}</p>
      )}
      {content}
    </div>
  );
}
