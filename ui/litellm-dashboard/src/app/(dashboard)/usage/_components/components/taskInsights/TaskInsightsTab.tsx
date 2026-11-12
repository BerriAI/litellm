"use client";

import { Grid3x3, Map as MapIcon, ListOrdered, Sparkles } from "lucide-react";
import { useQueryStates } from "nuqs";
import React from "react";
import { useTranslation } from "react-i18next";

import type { Team } from "@/components/key_team_helpers/key_list";
import { ErrorState } from "@/components/shared/ErrorState";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { extractErrorMessage } from "@/utils/errorUtils";
import { usageUrlParsers } from "../../usageUrlState";
import { Panel } from "../overview/Primitives";
import TaskBreakdown from "./TaskBreakdown";
import TaskClusterMap from "./TaskClusterMap";
import TaskDetailSheet from "./TaskDetailSheet";
import TaskHeatmap from "./TaskHeatmap";
import TaxonomyReview from "./TaxonomyReview";
import { coverage, formatCount, formatShare, formatUsdTotal, isDiscoveryBusy, taskColors } from "./taskInsightsData";
import {
  useEditDraftTask,
  useFreezeTaxonomy,
  useRequestDiscovery,
  useTaskInsightsMatrix,
  useTaskInsightsOverview,
  useTaskInsightsTask,
  useTaskInsightsTaxonomy,
} from "./useTaskInsights";

const ALL_TEAMS = "__all__";

interface TaskInsightsTabProps {
  startTime: Date | null;
  endTime: Date | null;
  teams: readonly Team[];
  canEdit: boolean;
  active: boolean;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex min-w-36 flex-col rounded-lg border bg-card px-4 py-3">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-lg font-semibold tabular-nums">{value}</span>
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </div>
  );
}

export default function TaskInsightsTab({ startTime, endTime, teams, canEdit, active }: TaskInsightsTabProps) {
  const { t } = useTranslation();
  const [params, setParams] = useQueryStates(usageUrlParsers);
  const filter = { startTime, endTime, teamId: params.ti_team };
  const overview = useTaskInsightsOverview(filter, active);
  const matrix = useTaskInsightsMatrix(filter, active);
  const detail = useTaskInsightsTask(filter, active ? params.ti_task : null, params.ti_model);
  const status = overview.data?.status;
  const taxonomy = useTaskInsightsTaxonomy(active && (status?.draft_task_count ?? 0) > 0);
  const discovery = useRequestDiscovery();
  const editDraft = useEditDraftTask();
  const freeze = useFreezeTaxonomy();
  const metric = params.ti_metric ?? "share";
  const tasks = overview.data?.tasks ?? [];
  const colors = React.useMemo(() => taskColors(tasks), [tasks]);
  const teamItems = React.useMemo(
    () => [
      { value: ALL_TEAMS, label: t("usage.taskInsights.allTeams") },
      ...teams.map((team) => ({ value: team.team_id, label: team.team_alias ?? team.team_id })),
    ],
    [teams, t],
  );

  const selectTask = (taskId: string) =>
    void setParams({ ti_task: params.ti_task === taskId && params.ti_model === null ? null : taskId, ti_model: null });
  const selectCell = (taskId: string, modelConfigId: string) => void setParams({ ti_task: taskId, ti_model: modelConfigId });

  if (overview.isError) {
    return (
      <ErrorState
        title={t("usage.taskInsights.loadFailed")}
        message={extractErrorMessage(overview.error)}
        onRetry={() => void overview.refetch()}
      />
    );
  }
  if (overview.isLoading || !overview.data) {
    return (
      <div role="status" aria-label={t("usage.taskInsights.loading")} className="grid gap-3 lg:grid-cols-2">
        <Skeleton className="h-80 w-full" />
        <Skeleton className="h-80 w-full" />
        <Skeleton className="h-64 w-full lg:col-span-2" />
      </div>
    );
  }

  const data = overview.data;
  const busy = isDiscoveryBusy(data.status);
  const drafts = (taxonomy.data?.definitions ?? []).filter((task) => task.status === "draft");
  const covered = coverage(data.totals);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <div className="w-56">
            <Select
              items={teamItems}
              value={params.ti_team ?? ALL_TEAMS}
              onValueChange={(value) => void setParams({ ti_team: value === ALL_TEAMS || value === null ? null : value })}
            >
              <SelectTrigger aria-label={t("usage.taskInsights.teamFilter")} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {teamItems.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {data.status.sample_data && <Badge variant="outline">{t("usage.taskInsights.sampleBadge")}</Badge>}
          {busy && <Badge variant="secondary">{t("usage.taskInsights.discoveryRunning")}</Badge>}
        </div>
        {canEdit && (
          <AlertDialog>
            <AlertDialogTrigger render={<Button variant="outline" size="sm" disabled={busy || discovery.isPending} />}>
              <Sparkles />
              {t("usage.taskInsights.discover")}
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>{t("usage.taskInsights.discoverTitle")}</AlertDialogTitle>
                <AlertDialogDescription>{t("usage.taskInsights.discoverDescription")}</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>{t("usage.taskInsights.cancel")}</AlertDialogCancel>
                <AlertDialogAction onClick={() => discovery.mutate(filter)}>
                  {t("usage.taskInsights.discoverConfirm")}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>

      {data.status.discovery_status === "failed" && data.status.discovery_error && (
        <ErrorState title={t("usage.taskInsights.discoveryFailed")} message={data.status.discovery_error} />
      )}

      {drafts.length > 0 && (
        <Panel title={t("usage.taskInsights.review.title")} subtitle={t("usage.taskInsights.review.subtitle")}>
          <TaxonomyReview
            drafts={drafts}
            canEdit={canEdit}
            saving={editDraft.isPending}
            freezing={freeze.isPending}
            onSave={(taskId, edit) => editDraft.mutate({ taskId, edit })}
            onFreeze={() => freeze.mutate(drafts.length)}
          />
        </Panel>
      )}

      {data.status.frozen_task_count === 0 ? (
        <Empty className="rounded-xl border">
          <EmptyHeader>
            <EmptyTitle>{t("usage.taskInsights.emptyTitle")}</EmptyTitle>
            <EmptyDescription>
              {canEdit ? t("usage.taskInsights.emptyDescriptionAdmin") : t("usage.taskInsights.emptyDescription")}
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <>
          <div className="flex flex-wrap gap-2">
            <Stat
              label={t("usage.taskInsights.stats.requests")}
              value={formatCount(data.totals.request_count)}
              hint={t("usage.taskInsights.stats.deliveries", { deliveries: formatCount(data.totals.delivery_count) })}
            />
            <Stat label={t("usage.taskInsights.stats.sessions")} value={formatCount(data.totals.session_count)} />
            <Stat
              label={t("usage.taskInsights.stats.coverage")}
              value={formatShare(covered) ?? t("usage.taskInsights.noData")}
              hint={t("usage.taskInsights.stats.coverageHint", {
                unknown: formatCount(data.totals.unknown_count),
                unclassified: formatCount(data.totals.unclassified_count),
              })}
            />
            <Stat
              label={t("usage.taskInsights.stats.spend")}
              value={formatUsdTotal(data.totals.spend) ?? t("usage.taskInsights.noData")}
              hint={t("usage.taskInsights.breakdown.costKnownFor", {
                known: formatCount(data.totals.cost_observations),
                total: formatCount(data.totals.request_count),
              })}
            />
            {data.totals.analysis_request_count > 0 && (
              <Stat
                label={t("usage.taskInsights.stats.analysis")}
                value={formatUsdTotal(data.totals.analysis_spend) ?? t("usage.taskInsights.noData")}
                hint={t("usage.taskInsights.stats.analysisHint", {
                  requests: formatCount(data.totals.analysis_request_count),
                })}
              />
            )}
          </div>
          <div className="grid gap-3 xl:grid-cols-5">
            <Panel
              icon={MapIcon}
              className="xl:col-span-3"
              title={t("usage.taskInsights.map.title")}
              subtitle={t("usage.taskInsights.map.subtitle")}
            >
              <TaskClusterMap
                tasks={tasks}
                points={data.points}
                colors={colors}
                selectedTaskId={params.ti_task}
                onSelect={selectTask}
              />
            </Panel>
            <Panel
              icon={ListOrdered}
              className="xl:col-span-2"
              title={t("usage.taskInsights.breakdown.title")}
              subtitle={t("usage.taskInsights.breakdown.subtitle")}
            >
              <TaskBreakdown tasks={tasks} colors={colors} selectedTaskId={params.ti_task} onSelect={selectTask} />
            </Panel>
          </div>
          <Panel
            icon={Grid3x3}
            title={t("usage.taskInsights.heatmap.title")}
            subtitle={t("usage.taskInsights.heatmap.subtitle")}
          >
            {matrix.isError ? (
              <ErrorState
                title={t("usage.taskInsights.heatmap.loadFailed")}
                message={extractErrorMessage(matrix.error)}
                onRetry={() => void matrix.refetch()}
              />
            ) : matrix.data ? (
              <TaskHeatmap
                matrix={matrix.data}
                metric={metric}
                onMetricChange={(next) => void setParams({ ti_metric: next === "share" ? null : next })}
                selectedTaskId={params.ti_task}
                selectedModelId={params.ti_model}
                onSelectCell={selectCell}
              />
            ) : (
              <Skeleton role="status" aria-label={t("usage.taskInsights.loading")} className="h-64 w-full" />
            )}
          </Panel>
          <p className="text-xs text-muted-foreground">
            {t("usage.taskInsights.footer", {
              start: data.window.start_date,
              end: data.window.end_date,
              refreshed: data.last_refreshed_at
                ? new Date(data.last_refreshed_at).toLocaleString()
                : t("usage.taskInsights.never"),
            })}
          </p>
        </>
      )}
      <TaskDetailSheet
        open={params.ti_task !== null}
        detail={detail.data}
        loading={detail.isLoading}
        error={detail.isError ? extractErrorMessage(detail.error) : null}
        sampleData={data.status.sample_data}
        onRetry={() => void detail.refetch()}
        onClose={() => void setParams({ ti_task: null, ti_model: null })}
        onClearModel={() => void setParams({ ti_model: null })}
      />
    </div>
  );
}
