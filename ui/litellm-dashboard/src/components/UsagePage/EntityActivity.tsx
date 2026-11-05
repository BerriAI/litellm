import { useCallback, useMemo, useState } from "react";
import { useQueryStates } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useAggregatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/useAggregatedDailyActivity";
import { ENTITY_API } from "@/app/(dashboard)/usage/_components/components/EntityUsage/entityFetchFns";
import { ModelSection } from "@/components/activity_metrics";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { hasProxyWideSpendView, spendScopeUserId } from "@/utils/roles";
import type { ActivityScope } from "@/utils/entityLinks";
import type { DailyActivityRequest } from "./dailyActivityApi";
import { boundedActivityMetrics } from "./keyActivityData";
import { activityPresets, defaultActivityDays, entityActivityParsers, localActivityDay } from "./entityActivityState";

interface Props {
  scope: ActivityScope;
  id: string;
  teamId?: string;
}

export default function EntityActivity({ scope, id, teamId }: Props) {
  const { t } = useTranslation();
  const { accessToken, userId, userRole } = useAuthorized();
  const [params, setParams] = useQueryStates(entityActivityParsers);
  const [defaults] = useState(defaultActivityDays);
  const from = params.activity_from ?? defaults.from;
  const to = params.activity_to ?? defaults.to;
  const invalidRange = from > to;
  const allowed = scope !== "model" || hasProxyWideSpendView(userRole);
  const currentUser = spendScopeUserId(userRole, userId);
  const keyTeamId = scope === "key" && currentUser ? teamId : undefined;
  const entity = scope === "team" || keyTeamId ? "team" : "user";
  const request = useMemo<DailyActivityRequest>(() => {
    let entityIds = currentUser ? [currentUser] : null;
    if (scope === "team" || scope === "user") entityIds = [id];
    if (keyTeamId) entityIds = [keyTeamId];
    return {
      accessToken: accessToken ?? "",
      startTime: new Date(`${from}T00:00:00`),
      endTime: new Date(`${to}T23:59:59`),
      timezoneOffsetMinutes: 0,
      entityIds,
      apiKey: scope === "key" ? id : null,
      model: scope === "model" ? id : null,
      bounded: true,
      includeModelBreakdown: scope === "key",
      includeCurrentUtcDay: true,
    };
  }, [accessToken, currentUser, scope, id, keyTeamId, from, to]);
  const hasScope = Boolean(accessToken && id);
  const { data, loading, failed, reload } = useAggregatedDailyActivity({
    fetch: () => ENTITY_API[entity].aggregated(request),
    enabled: hasScope && allowed && !invalidRange,
    deps: [accessToken, currentUser, scope, id, keyTeamId, from, to],
  });

  const fetchTopApiKeys = useCallback(
    (model: string) => ENTITY_API.user.modelTopKeys(request, model, false),
    [request],
  );
  const { metrics, otherModels } = boundedActivityMetrics(data);

  if (!allowed) return <p className="text-sm text-muted-foreground">{t("entityActivity.modelRequiresGlobal")}</p>;

  function renderActivity() {
    if (invalidRange)
      return (
        <p role="alert" className="text-sm text-destructive">
          {t("entityActivity.invalidRange")}
        </p>
      );
    if (loading)
      return (
        <p role="status" className="text-sm text-muted-foreground">
          {t("entityActivity.loading")}
        </p>
      );
    if (failed)
      return (
        <div role="alert" className="flex items-center gap-2 text-sm">
          <span>{t("entityActivity.failed")}</span>
          <Button variant="outline" size="sm" onClick={reload}>
            {t("entityActivity.retry")}
          </Button>
        </div>
      );
    if (data.results.length === 0) return <p className="text-sm text-muted-foreground">{t("entityActivity.empty")}</p>;
    return (
      <ModelSection
        modelName={id}
        metrics={metrics}
        otherModels={otherModels}
        fetchTopApiKeys={scope === "model" ? fetchTopApiKeys : undefined}
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end gap-2">
        <span className="text-sm text-muted-foreground">UTC</span>
        <AdvancedDatePicker
          compact
          label=""
          className="max-w-xs"
          presets={activityPresets}
          showTimeRange={false}
          value={{ from: request.startTime, to: request.endTime }}
          onValueChange={(range) => {
            if (range.from && range.to) {
              void setParams({ activity_from: localActivityDay(range.from), activity_to: localActivityDay(range.to) });
            }
          }}
        />
      </div>
      {renderActivity()}
    </div>
  );
}
