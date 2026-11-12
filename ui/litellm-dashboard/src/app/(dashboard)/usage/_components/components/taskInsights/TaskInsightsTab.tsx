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
import { usageUrlParsers, type TaskInsightsMetric } from "../../usageUrlState";
import { Panel } from "../overview/Primitives";
import TaskBreakdown from "./TaskBreakdown";
import TaskClusterMap from "./TaskClusterMap";
import TaskDetailSheet from "./TaskDetailSheet";
import TaskHeatmap from "./TaskHeatmap";
import TaxonomyReview from "./TaxonomyReview";
import {
  coverage,
  formatCount,
  formatShare,
  formatUsdTotal,
  isDiscoveryBusy,
  taskColors,
  type TaskInsightsOverview,
} from "./taskInsightsData";
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

function MatrixBody({
  matrix,
  metric,
  selectedTaskId,
  selectedModelId,
  onMetricChange,
  onSelectCell,
}: {
  matrix: ReturnType<typeof useTaskInsightsMatrix>;
  metric: TaskInsightsMetric;
  selectedTaskId: string | null;
  selectedModelId: string | null;
  onMetricChange: (metric: TaskInsightsMetric) => void;
  onSelectCell: (taskId: string, modelConfigId: string) => void;
}) {
  const { t } = useTranslation();
  if (matrix.isError) {
    return (
      <ErrorState
        title={t("usage.taskInsights.heatmap.loadFailed")}
        message={extractErrorMessage(matrix.error)}
        onRetry={() => void matrix.refetch()}
      />
    );
  }
  if (!matrix.data) {
    return <Skeleton role="status" aria-label={t("usage.taskInsights.loading")} className="h-64 w-full" />;
  }
  return (
    <TaskHeatmap
      matrix={matrix.data}
      metric={metric}
      onMetricChange={onMetricChange}
      selectedTaskId={selectedTaskId}
      selectedModelId={selectedModelId}
      onSelectCell={onSelectCell}
    />
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div role="group" aria-label={label} className="flex min-w-36 flex-col rounded-lg border bg-card px-4 py-3">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-lg font-semibold tabular-nums">{value}</span>
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </div>
  );
}

function TaskInsightsToolbar({
  teams,
  teamId,
  onTeamChange,
  sampleData,
  busy,
  canEdit,
  discovering,
  onDiscover,
}: {
  teams: readonly Team[];
  teamId: string | null;
  onTeamChange: (teamId: string | null) => void;
  sampleData: boolean;
  busy: boolean;
  canEdit: boolean;
  discovering: boolean;
  onDiscover: () => void;
}) {
  const { t } = useTranslation();
  const teamItems = React.useMemo(
    () => [
      { value: ALL_TEAMS, label: t("usage.taskInsights.allTeams") },
      ...teams.map((team) => ({ value: team.team_id, label: team.team_alias ?? team.team_id })),
    ],
    [teams, t],
  );
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <div className="w-56">
          <Select
            items={teamItems}
            value={teamId ?? ALL_TEAMS}
            onValueChange={(value) => onTeamChange(value === ALL_TEAMS || value === null ? null : value)}
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
        {sampleData && <Badge variant="outline">{t("usage.taskInsights.sampleBadge")}</Badge>}
        {busy && <Badge variant="secondary">{t("usage.taskInsights.discoveryRunning")}</Badge>}
      </div>
      {canEdit && (
        <AlertDialog>
          <AlertDialogTrigger render={<Button variant="outline" size="sm" disabled={busy || discovering} />}>
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
              <AlertDialogAction onClick={onDiscover}>{t("usage.taskInsights.discoverConfirm")}</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
}

function TaskInsightsStats({ totals }: { totals: TaskInsightsOverview["totals"] }) {
  const { t } = useTranslation();
  const covered = coverage(totals);
  return (
    <div className="flex flex-wrap gap-2">
      <Stat
        label={t("usage.taskInsights.stats.requests")}
        value={formatCount(totals.request_count)}
        hint={t("usage.taskInsights.stats.deliveries", { deliveries: formatCount(totals.delivery_count) })}
      />
      <Stat label={t("usage.taskInsights.stats.sessions")} value={formatCount(totals.session_count)} />
      <Stat
        label={t("usage.taskInsights.stats.coverage")}
        value={formatShare(covered) ?? t("usage.taskInsights.noData")}
        hint={t("usage.taskInsights.stats.coverageHint", {
          unknown: formatCount(totals.unknown_count),
          unclassified: formatCount(totals.unclassified_count),
        })}
      />
      <Stat
        label={t("usage.taskInsights.stats.spend")}
        value={formatUsdTotal(totals.spend) ?? t("usage.taskInsights.noData")}
        hint={t("usage.taskInsights.breakdown.costKnownFor", {
          known: formatCount(totals.cost_observations),
          total: formatCount(totals.request_count),
        })}
      />
      {totals.analysis_request_count > 0 && (
        <Stat
          label={t("usage.taskInsights.stats.analysis")}
          value={formatUsdTotal(totals.analysis_spend) ?? t("usage.taskInsights.noData")}
          hint={t("usage.taskInsights.stats.analysisHint", {
            requests: formatCount(totals.analysis_request_count),
          })}
        />
      )}
    </div>
  );
}

function TaskInsightsContent({
  data,
  tasks,
  colors,
  matrix,
  metric,
  selectedTaskId,
  selectedModelId,
  canEdit,
  onSelectTask,
  onSelectCell,
  onMetricChange,
}: {
  data: TaskInsightsOverview;
  tasks: TaskInsightsOverview["tasks"];
  colors: ReadonlyMap<string, string>;
  matrix: ReturnType<typeof useTaskInsightsMatrix>;
  metric: TaskInsightsMetric;
  selectedTaskId: string | null;
  selectedModelId: string | null;
  canEdit: boolean;
  onSelectTask: (taskId: string) => void;
  onSelectCell: (taskId: string, modelConfigId: string) => void;
  onMetricChange: (metric: TaskInsightsMetric) => void;
}) {
  const { t } = useTranslation();
  if (data.status.frozen_task_count === 0) {
    return (
      <Empty className="rounded-xl border">
        <EmptyHeader>
          <EmptyTitle>{t("usage.taskInsights.emptyTitle")}</EmptyTitle>
          <EmptyDescription>
            {canEdit ? t("usage.taskInsights.emptyDescriptionAdmin") : t("usage.taskInsights.emptyDescription")}
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }
  const refreshed = data.last_refreshed_at
    ? new Date(data.last_refreshed_at).toLocaleString()
    : t("usage.taskInsights.never");
  return (
    <>
      <TaskInsightsStats totals={data.totals} />
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
            selectedTaskId={selectedTaskId}
            onSelect={onSelectTask}
          />
        </Panel>
        <Panel
          icon={ListOrdered}
          className="xl:col-span-2"
          title={t("usage.taskInsights.breakdown.title")}
          subtitle={t("usage.taskInsights.breakdown.subtitle")}
        >
          <TaskBreakdown tasks={tasks} colors={colors} selectedTaskId={selectedTaskId} onSelect={onSelectTask} />
        </Panel>
      </div>
      <Panel
        icon={Grid3x3}
        title={t("usage.taskInsights.heatmap.title")}
        subtitle={t("usage.taskInsights.heatmap.subtitle")}
      >
        <MatrixBody
          matrix={matrix}
          metric={metric}
          selectedTaskId={selectedTaskId}
          selectedModelId={selectedModelId}
          onMetricChange={onMetricChange}
          onSelectCell={onSelectCell}
        />
      </Panel>
      <p className="text-xs text-muted-foreground">
        {t("usage.taskInsights.footer", {
          start: data.window.start_date,
          end: data.window.end_date,
          refreshed,
        })}
      </p>
    </>
  );
}

function DraftReview({ draftCount, canEdit, active }: { draftCount: number; canEdit: boolean; active: boolean }) {
  const { t } = useTranslation();
  const taxonomy = useTaskInsightsTaxonomy(active && draftCount > 0);
  const editDraft = useEditDraftTask();
  const freeze = useFreezeTaxonomy();
  const drafts = (taxonomy.data?.definitions ?? []).filter((task) => task.status === "draft");
  if (drafts.length === 0) return null;
  return (
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
  );
}

export default function TaskInsightsTab({ startTime, endTime, teams, canEdit, active }: TaskInsightsTabProps) {
  const { t } = useTranslation();
  const [params, setParams] = useQueryStates(usageUrlParsers);
  const filter = { startTime, endTime, teamId: params.ti_team };
  const overview = useTaskInsightsOverview(filter, active);
  const matrix = useTaskInsightsMatrix(filter, active);
  const detail = useTaskInsightsTask(filter, active ? params.ti_task : null, params.ti_model);
  const discovery = useRequestDiscovery();
  const tasks = React.useMemo(() => overview.data?.tasks ?? [], [overview.data]);
  const colors = React.useMemo(() => taskColors(tasks), [tasks]);

  const selectTask = (taskId: string) =>
    void setParams({ ti_task: params.ti_task === taskId && params.ti_model === null ? null : taskId, ti_model: null });

  if (overview.isError) {
    return (
      <ErrorState
        title={t("usage.taskInsights.loadFailed")}
        message={extractErrorMessage(overview.error)}
        onRetry={() => void overview.refetch()}
      />
    );
  }
  if (!overview.data) {
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
  return (
    <div className="flex flex-col gap-3">
      <TaskInsightsToolbar
        teams={teams}
        teamId={params.ti_team}
        onTeamChange={(teamId) => void setParams({ ti_team: teamId })}
        sampleData={data.status.sample_data}
        busy={busy}
        canEdit={canEdit}
        discovering={discovery.isPending}
        onDiscover={() => discovery.mutate(filter)}
      />
      {data.status.discovery_error && data.status.discovery_status === "failed" && (
        <ErrorState title={t("usage.taskInsights.discoveryFailed")} message={data.status.discovery_error} />
      )}
      <DraftReview draftCount={data.status.draft_task_count} canEdit={canEdit} active={active} />
      <TaskInsightsContent
        data={data}
        tasks={tasks}
        colors={colors}
        matrix={matrix}
        metric={params.ti_metric ?? "share"}
        selectedTaskId={params.ti_task}
        selectedModelId={params.ti_model}
        canEdit={canEdit}
        onSelectTask={selectTask}
        onSelectCell={(taskId, modelConfigId) => void setParams({ ti_task: taskId, ti_model: modelConfigId })}
        onMetricChange={(next) => void setParams({ ti_metric: next === "share" ? null : next })}
      />
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
