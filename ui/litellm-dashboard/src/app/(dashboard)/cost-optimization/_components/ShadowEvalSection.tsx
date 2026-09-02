"use client";

import React, { useMemo, useState } from "react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CircleHelp } from "lucide-react";

import { Card } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError } from "@/lib/http/client";

import { usd } from "./costOptimizationUtils";
import { StartForm } from "./ShadowEvalStartForm";
import {
  useShadowEvalJob,
  useShadowEvalJobs,
  useStopShadowEval,
  type ShadowEvalJob,
  type ShadowEvalJobTarget,
  type ShadowEvalSlice,
} from "./useShadowEval";

const pct = (value: number): string => `${value.toFixed(1)}%`;

const MIN_TURNS_FOR_CONFIDENCE = 30;

type ShadowEvalDirection = ShadowEvalJob["direction"];

const otherArmLabel = (direction: ShadowEvalDirection): string =>
  direction === "reverse" ? "Baseline" : "Current model";

const routerWinRate = (direction: ShadowEvalDirection, slice: ShadowEvalSlice): number =>
  direction === "reverse" ? slice.real_win_rate_pct : slice.shadow_win_rate_pct;

const otherArmWinRate = (direction: ShadowEvalDirection, slice: ShadowEvalSlice): number =>
  direction === "reverse" ? slice.shadow_win_rate_pct : slice.real_win_rate_pct;

const routerArmSpend = (direction: ShadowEvalDirection, results: NonNullable<ShadowEvalJob["results"]>): number =>
  direction === "reverse" ? results.sampled_real_spend : results.sampled_shadow_spend;

const otherArmSpend = (direction: ShadowEvalDirection, results: NonNullable<ShadowEvalJob["results"]>): number =>
  direction === "reverse" ? results.sampled_shadow_spend : results.sampled_real_spend;

const routerSliceSpend = (direction: ShadowEvalDirection, slice: ShadowEvalSlice): number =>
  direction === "reverse" ? slice.real_spend : slice.shadow_spend;

const otherSliceSpend = (direction: ShadowEvalDirection, slice: ShadowEvalSlice): number =>
  direction === "reverse" ? slice.shadow_spend : slice.real_spend;

const routerMatchedOrBeatPct = (
  direction: ShadowEvalDirection,
  results: NonNullable<ShadowEvalJob["results"]>,
): number =>
  direction === "reverse"
    ? 100 - results.overall_shadow_win_rate_pct
    : results.overall_shadow_win_rate_pct + results.overall_tie_rate_pct;

export const shadowedTargetLabel = (target: ShadowEvalJobTarget): string =>
  target.target_alias ||
  target.key_name ||
  (target.target_type === "key" ? `${target.target_id.slice(0, 10)}…` : target.target_id);

const shadowedTargetsLabel = (job: ShadowEvalJob): string =>
  job.targets.length === 1 ? shadowedTargetLabel(job.targets[0]) : `${job.targets.length} targets`;

const totalBudget = (job: ShadowEvalJob): number | null =>
  job.targets.reduce<number | null>(
    (sum, target) => (sum === null || target.max_budget == null ? null : sum + target.max_budget),
    0,
  );

const totalSpend = (job: ShadowEvalJob): number => job.targets.reduce((sum, target) => sum + (target.spend ?? 0), 0);

const targetSpent = (target: ShadowEvalJobTarget): boolean => {
  const spendBudgetReached = target.max_budget != null && target.spend != null && target.spend >= target.max_budget;
  const turnValveReached = target.attempt_count != null && target.attempt_count >= target.max_turns;
  return spendBudgetReached || turnValveReached;
};

const targetStatus = (job: ShadowEvalJob, target: ShadowEvalJobTarget): string => {
  if (job.status === "completed" || (target.stopped_at == null && targetSpent(target))) return "completed";
  return target.stopped_at != null ? "stopped" : "running";
};

const jobRouters = (job: ShadowEvalJob): string => (job.router_names ?? [job.router_name]).join(", ");

const jobHeadline = (job: ShadowEvalJob): React.ReactNode =>
  job.direction === "reverse" ? (
    <>
      Comparing <span className="font-mono text-xs">{jobRouters(job)}</span> to{" "}
      <span className="font-mono text-xs">{job.baseline_model}</span> on {job.shadow_percentage}% of{" "}
      <span className="font-mono text-xs">{shadowedTargetsLabel(job)}</span> traffic
    </>
  ) : (
    <>
      Shadowing {job.shadow_percentage}% of <span className="font-mono text-xs">{shadowedTargetsLabel(job)}</span>{" "}
      traffic via <span className="font-mono text-xs">{jobRouters(job)}</span>
    </>
  );

const isActive = (job: ShadowEvalJob): boolean => job.status === "running";

const endsIn = (endsAt: string | null | undefined): string | null => {
  if (!endsAt) return null;
  const remainingMs = new Date(endsAt).getTime() - Date.now();
  if (!Number.isFinite(remainingMs)) return null;
  if (remainingMs <= 0) return "ending now";
  const days = Math.round(remainingMs / 86_400_000);
  return days >= 2 ? `ends in ${days} days` : "ends within a day";
};

const STATUS_STYLES: Record<string, string> = {
  running: "bg-info/10 text-info",
  completed: "bg-success/10 text-success",
  stopped: "bg-secondary text-muted-foreground",
};

const StatusBadge: React.FC<{ status: string }> = ({ status }) => (
  <Badge variant="secondary" className={STATUS_STYLES[status] ?? STATUS_STYLES.stopped}>
    {status}
  </Badge>
);

const SliceTable: React.FC<{
  groupHeader: string;
  direction: ShadowEvalDirection;
  slices: readonly ShadowEvalSlice[];
}> = ({ groupHeader, direction, slices }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>{groupHeader}</TableHead>
        {[
          "Judged turns",
          "Router wins",
          `${otherArmLabel(direction)} wins`,
          "Ties",
          "Judge confidence",
          "Router cost",
          `${otherArmLabel(direction)} cost`,
        ].map((label) => (
          <TableHead key={label} className="text-right">
            {label}
          </TableHead>
        ))}
      </TableRow>
    </TableHeader>
    <TableBody>
      {slices.map((slice) => (
        <TableRow key={slice.group}>
          <TableCell className="font-medium text-foreground">
            {slice.group}
            {slice.turn_count < MIN_TURNS_FOR_CONFIDENCE && (
              <span className="ml-2 text-xs font-normal text-muted-foreground">(low sample)</span>
            )}
          </TableCell>
          <TableCell className="text-right tabular-nums">{slice.turn_count.toLocaleString()}</TableCell>
          <TableCell className="text-right font-medium tabular-nums text-foreground">
            {pct(routerWinRate(direction, slice))}
          </TableCell>
          <TableCell className="text-right tabular-nums">{pct(otherArmWinRate(direction, slice))}</TableCell>
          <TableCell className="text-right tabular-nums">{pct(slice.tie_rate_pct)}</TableCell>
          <TableCell className="text-right tabular-nums">{slice.avg_judge_confidence.toFixed(2)}</TableCell>
          <TableCell className="text-right tabular-nums">
            {routerSliceSpend(direction, slice) > 0 ? usd(routerSliceSpend(direction, slice)) : "-"}
          </TableCell>
          <TableCell className="text-right tabular-nums">
            {otherSliceSpend(direction, slice) > 0 ? usd(otherSliceSpend(direction, slice)) : "-"}
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const CostComparison: React.FC<{
  direction: ShadowEvalDirection;
  results: NonNullable<ShadowEvalJob["results"]>;
}> = ({ direction, results }) => {
  const routerSpend = routerArmSpend(direction, results);
  const otherSpend = otherArmSpend(direction, results);
  if (routerSpend <= 0 || otherSpend <= 0) return null;
  const savingsPct = otherSpend > 0 ? ((otherSpend - routerSpend) / otherSpend) * 100 : null;
  const cacheHits = results.by_tier.reduce((sum, slice) => sum + slice.cache_hit_turns, 0);
  return (
    <div className="flex min-w-[240px] flex-1 flex-col gap-1 border-t px-6 py-4 sm:border-l sm:border-t-0">
      <p className="flex items-center gap-1 text-[11px] uppercase tracking-wide text-muted-foreground">
        Router cost vs {direction === "reverse" ? "the baseline" : "your current model"}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help" />} />
            <TooltipContent>
              Each arm is priced as its completion plus its own routing classifier call, measured on the same judged
              turns; the judge&apos;s cost is excluded from both arms
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </p>
      <p
        className={`text-3xl font-semibold ${savingsPct != null && savingsPct > 0 ? "text-success" : "text-foreground"}`}
      >
        {savingsPct != null ? `${savingsPct > 0 ? "-" : "+"}${Math.abs(savingsPct).toFixed(1)}%` : "n/a"}
      </p>
      <p className="text-xs text-muted-foreground">
        {usd(routerSpend)} vs {usd(otherSpend)} on the same judged turns
        {cacheHits > 0 ? `; ${cacheHits.toLocaleString()} cache-served turns excluded` : ""}
      </p>
    </div>
  );
};

const VerdictBar: React.FC<{ direction: ShadowEvalDirection; results: NonNullable<ShadowEvalJob["results"]> }> = ({
  direction,
  results,
}) => {
  const ties = results.overall_tie_rate_pct;
  const routerWins =
    direction === "reverse"
      ? Math.max(0, 100 - results.overall_shadow_win_rate_pct - ties)
      : results.overall_shadow_win_rate_pct;
  const segments = [
    { label: "Router won", value: routerWins, fill: "bg-success" },
    { label: "Tie", value: ties, fill: "bg-success/20" },
    {
      label: `${otherArmLabel(direction)} won`,
      value: Math.max(0, 100 - routerWins - ties),
      fill: "bg-muted-foreground/30",
    },
  ];
  return (
    <div className="space-y-2 border-b px-6 py-4">
      <div className="flex h-2 w-full overflow-hidden rounded-full" role="img" aria-label="Verdict breakdown">
        {segments
          .filter((segment) => segment.value > 0)
          .map((segment) => (
            <div key={segment.label} className={segment.fill} style={{ width: `${segment.value}%` }} />
          ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {segments.map((segment) => (
          <span key={segment.label} className="flex items-center gap-1.5">
            <span className={`size-2 rounded-full ${segment.fill}`} />
            {segment.label} {pct(segment.value)}
          </span>
        ))}
      </div>
    </div>
  );
};

const TargetTable: React.FC<{ job: ShadowEvalJob }> = ({ job }) => {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Target</TableHead>
          <TableHead>Status</TableHead>
          {["Budget used", "Router wins", `${otherArmLabel(job.direction)} wins`].map((label) => (
            <TableHead key={label} className="text-right">
              {label}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {job.targets.map((target) => {
          const slice = target.verdicts;
          return (
            <TableRow key={`${target.target_type}:${target.target_id}`}>
              <TableCell className="font-medium text-foreground">
                {shadowedTargetLabel(target)}
                {target.target_type !== "key" && (
                  <span className="ml-2 text-xs font-normal text-muted-foreground">{target.target_type}</span>
                )}
              </TableCell>
              <TableCell>
                <StatusBadge status={targetStatus(job, target)} />
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {target.max_budget != null
                  ? `${usd(target.spend ?? 0)} / ${usd(target.max_budget)}`
                  : `${(target.attempt_count ?? slice?.turn_count ?? 0).toLocaleString()} / ${target.max_turns.toLocaleString()} turns`}
              </TableCell>
              {slice ? (
                <>
                  <TableCell className="text-right font-medium tabular-nums text-foreground">
                    {pct(routerWinRate(job.direction, slice))}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {pct(otherArmWinRate(job.direction, slice))}
                  </TableCell>
                </>
              ) : (
                <TableCell colSpan={2} className="text-right text-muted-foreground">
                  No verdicts yet
                </TableCell>
              )}
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
};

const emptyResultsText = (job: ShadowEvalJob, resultsError: boolean): string => {
  if (resultsError) return "Results could not be loaded. Retrying.";
  if (isActive(job)) return "Collecting verdicts. Results appear as sampled requests are judged.";
  if (job.judged_count === 0) return "No verdicts were recorded for this job.";
  return "Loading results...";
};

const ResultsBody: React.FC<{ job: ShadowEvalJob; resultsError?: boolean }> = ({ job, resultsError = false }) => {
  const results = job.results;
  const hasVerdicts = results != null && (results.by_tier.length > 0 || results.by_current_model.length > 0);
  return (
    <>
      {job.targets.length > 1 && (
        <div className="border-b">
          <TargetTable job={job} />
        </div>
      )}
      {/* results == null re-stated for TS narrowing; hasVerdicts alone cannot narrow it */}
      {!hasVerdicts || results == null ? (
        <p className="px-6 py-8 text-center text-sm text-muted-foreground">{emptyResultsText(job, resultsError)}</p>
      ) : (
        <>
          <div className="flex flex-wrap border-b">
            <div className="flex min-w-[240px] flex-1 flex-col gap-1 px-6 py-4">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Router matched or beat {job.direction === "reverse" ? "the baseline" : "your current model"}
              </p>
              <p className="text-3xl font-semibold text-foreground">
                {pct(routerMatchedOrBeatPct(job.direction, results))}
              </p>
              <p className="text-xs text-muted-foreground">
                of {(job.judged_count ?? 0).toLocaleString()} judged responses
              </p>
            </div>
            <CostComparison direction={job.direction} results={results} />
          </div>
          <VerdictBar direction={job.direction} results={results} />
          {(results.by_router ?? []).length > 1 && (
            <div className="border-b">
              <SliceTable groupHeader="Router" direction={job.direction} slices={results.by_router ?? []} />
            </div>
          )}
          {results.by_current_model.length > 0 && (
            <SliceTable
              groupHeader={job.direction === "reverse" ? "Router pick" : "Compared against"}
              direction={job.direction}
              slices={results.by_current_model}
            />
          )}
          {results.by_tier.length > 0 && (
            <div className={results.by_current_model.length > 0 ? "border-t" : ""}>
              <SliceTable groupHeader="Prompt difficulty" direction={job.direction} slices={results.by_tier} />
            </div>
          )}
        </>
      )}
    </>
  );
};

const JobResults: React.FC<{
  job: ShadowEvalJob;
  onStop: () => void;
  stopPending: boolean;
  resultsError?: boolean;
  readOnly?: boolean;
}> = ({ job, onStop, stopPending, resultsError = false, readOnly = false }) => {
  const active = isActive(job);
  const remaining = endsIn(job.ends_at);
  return (
    <Card className="overflow-hidden py-0">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-4">
        <div className="flex items-center gap-3">
          <StatusBadge status={job.status} />
          <div>
            <p className="text-sm font-medium text-foreground">{jobHeadline(job)}</p>
            <p className="text-xs text-muted-foreground">
              {(job.judged_count ?? 0).toLocaleString()} turns judged · {(job.error_count ?? 0).toLocaleString()}{" "}
              errored · {usd(totalSpend(job))}
              {totalBudget(job) !== null ? ` of ${usd(totalBudget(job) ?? 0)}` : ""} eval spend
              {active && remaining ? ` · ${remaining}` : ""}
            </p>
          </div>
        </div>
        {active && !readOnly && (
          <Button variant="outline" size="sm" onClick={onStop} disabled={stopPending}>
            {stopPending ? "Stopping..." : "Stop"}
          </Button>
        )}
      </div>
      {(job.error_count ?? 0) > 0 && job.last_error != null && (
        <p className="border-b bg-destructive/10 px-6 py-2 text-xs text-destructive">
          Last failure: <span className="font-mono">{job.last_error}</span>
        </p>
      )}
      <ResultsBody job={job} resultsError={resultsError} />
    </Card>
  );
};

const previousSummary = (job: ShadowEvalJob): string => {
  const results = job.results;
  if (results) return pct(routerMatchedOrBeatPct(job.direction, results));
  return job.judged_count === 0 ? "no verdicts" : "view results";
};

const PreviousJob: React.FC<{ job: ShadowEvalJob }> = ({ job }) => {
  const [expanded, setExpanded] = useState(false);
  const { data: detail, isError } = useShadowEvalJob(expanded ? job.job_id : null);
  const shown = detail ?? job;
  return (
    <div className="border-b last:border-b-0">
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((open) => !open)}
        className="flex w-full flex-wrap items-center justify-between gap-3 px-6 py-3 text-left hover:bg-muted/50"
      >
        <div className="flex items-center gap-3">
          <StatusBadge status={shown.status} />
          <div>
            <p className="text-sm font-medium text-foreground">{jobHeadline(shown)}</p>
            <p className="text-xs text-muted-foreground">
              {shown.judged_count != null &&
                `${shown.judged_count.toLocaleString()} judged · ${(shown.error_count ?? 0).toLocaleString()} errored · ${usd(totalSpend(shown))} eval spend · `}
              {new Date(shown.created_at).toLocaleDateString()}
            </p>
          </div>
        </div>
        <span className="text-sm font-medium text-foreground">{previousSummary(shown)}</span>
      </button>
      {expanded && (
        <div className="border-t">
          <ResultsBody job={shown} resultsError={isError} />
        </div>
      )}
    </div>
  );
};

const PreviousJobs: React.FC<{ jobs: readonly ShadowEvalJob[] }> = ({ jobs }) => {
  const [open, setOpen] = useState(false);
  if (jobs.length === 0) return null;
  return (
    <Card className="overflow-hidden py-0">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center justify-between gap-3 px-6 py-3 text-left hover:bg-muted/50"
      >
        <span className="text-sm font-medium text-foreground">Previous evaluations ({jobs.length})</span>
        <span className="text-xs text-muted-foreground">{open ? "Hide" : "Show"}</span>
      </button>
      {open && (
        <div className="border-t">
          {jobs.map((job) => (
            <PreviousJob key={job.job_id} job={job} />
          ))}
        </div>
      )}
    </Card>
  );
};

const JobCard: React.FC<{ job: ShadowEvalJob; readOnly: boolean }> = ({ job, readOnly }) => {
  const { data: detail, isError } = useShadowEvalJob(job.job_id);
  const stop = useStopShadowEval();
  const shown = detail ?? job;
  return (
    <JobResults
      job={shown}
      onStop={() => stop.mutate(shown.job_id)}
      stopPending={stop.isPending}
      resultsError={isError}
      readOnly={readOnly}
    />
  );
};

const ShadowEvalSection: React.FC = () => {
  const { data: jobs, error, isPending } = useShadowEvalJobs();
  const { isViewOnly } = useAuthorized();
  const { showcased, listed } = useMemo(() => {
    const active = (jobs ?? []).filter(isActive);
    const finished = (jobs ?? []).filter((job) => !isActive(job));
    const shown = active.length > 0 ? active : finished.slice(0, 1);
    return { showcased: shown, listed: finished.filter((job) => !shown.includes(job)) };
  }, [jobs]);

  if (error instanceof ApiError && error.status === 403) return null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-2">
        <h2 className="text-xl font-semibold text-foreground">Shadow eval</h2>
        <p className="text-sm text-muted-foreground">
          Blind-judge the auto-router on the real traffic of a key, team, or user (teams and users cover
          JWT-authenticated traffic): against the models they use today before switching, or against a fixed baseline
          after they have switched.
        </p>
      </div>

      {error != null && (
        <p className="text-sm text-destructive">Existing evaluations could not be loaded. Refresh the page to retry.</p>
      )}

      {isPending && error == null && <p className="text-sm text-muted-foreground">Loading evaluations...</p>}

      {showcased.map((job) => (
        <JobCard key={job.job_id} job={job} readOnly={isViewOnly} />
      ))}

      {!isViewOnly && <StartForm />}

      <PreviousJobs jobs={listed} />
    </div>
  );
};

export default ShadowEvalSection;
