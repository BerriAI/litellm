"use client";

import React, { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError } from "@/lib/http/client";

import { usd } from "./costOptimizationUtils";
import {
  useShadowEvalJob,
  useShadowEvalJobs,
  useStartShadowEval,
  useStopShadowEval,
  type ShadowEvalJob,
  type ShadowEvalTierResult,
} from "./useShadowEval";

const pct = (value: number): string => `${value.toFixed(1)}%`;

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-secondary text-muted-foreground",
  running: "bg-blue-50 text-blue-700",
  completed: "bg-emerald-50 text-emerald-700",
  failed: "bg-red-50 text-destructive",
};

const StatusBadge: React.FC<{ status: string }> = ({ status }) => (
  <Badge variant="secondary" className={STATUS_STYLES[status] ?? STATUS_STYLES.pending}>
    {status}
  </Badge>
);

/** Verdict counts are meaningless below this; the table warns instead of misleading. */
const MIN_TURNS_FOR_CONFIDENCE = 30;

const TierResultsTable: React.FC<{ groups: readonly ShadowEvalTierResult[] }> = ({ groups }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Router tier</TableHead>
        <TableHead className="text-right">Judged turns</TableHead>
        <TableHead className="text-right">Router pick wins</TableHead>
        <TableHead className="text-right">Current model wins</TableHead>
        <TableHead className="text-right">Ties</TableHead>
        <TableHead className="text-right">Judge confidence</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {groups.map((g) => (
        <TableRow key={g.tier}>
          <TableCell className="font-medium text-foreground">
            {g.tier}
            {g.turn_count < MIN_TURNS_FOR_CONFIDENCE ? (
              <span className="ml-2 text-xs text-muted-foreground">(low sample)</span>
            ) : null}
          </TableCell>
          <TableCell className="text-right tabular-nums">{g.turn_count.toLocaleString()}</TableCell>
          <TableCell className="text-right font-medium tabular-nums text-foreground">
            {pct(g.shadow_win_rate_pct)}
          </TableCell>
          <TableCell className="text-right tabular-nums">{pct(g.real_win_rate_pct)}</TableCell>
          <TableCell className="text-right tabular-nums">{pct(g.tie_rate_pct)}</TableCell>
          <TableCell className="text-right tabular-nums">{g.avg_judge_confidence.toFixed(2)}</TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const JobResults: React.FC<{
  job: ShadowEvalJob;
  onStop: () => void;
  stopPending: boolean;
}> = ({ job, onStop, stopPending }) => {
  const active = job.status === "pending" || job.status === "running";
  const results = job.results;
  const okOrBetter = results ? results.overall_shadow_win_rate_pct + results.overall_tie_rate_pct : null;
  return (
    <Card className="overflow-hidden py-0">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-4">
        <div className="flex items-center gap-3">
          <StatusBadge status={job.status} />
          <div>
            <p className="text-sm font-medium text-foreground">
              Shadowing {job.shadow_percentage}% via <span className="font-mono text-xs">{job.router_name}</span>
            </p>
            <p className="text-xs text-muted-foreground">
              {job.completed_count.toLocaleString()} judged · {job.failed_count.toLocaleString()} failed ·{" "}
              {job.cost_actual != null ? `${usd(job.cost_actual)} judge spend` : "no judge spend yet"}
              {job.cost_estimate != null ? ` (est. ${usd(job.cost_estimate)}/wk)` : ""}
            </p>
          </div>
        </div>
        {active ? (
          <Button variant="outline" size="sm" onClick={onStop} disabled={stopPending}>
            {stopPending ? "Stopping…" : "Stop"}
          </Button>
        ) : null}
      </div>

      {results && results.groups.length > 0 ? (
        <>
          <div className="grid gap-0 border-b sm:grid-cols-2">
            <div className="flex flex-col justify-center gap-1 px-6 py-4">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Router pick judged as good or better
              </p>
              <p className="text-3xl font-semibold text-foreground">{okOrBetter != null ? pct(okOrBetter) : "—"}</p>
            </div>
            <div className="flex flex-col justify-center gap-1 border-t px-6 py-4 sm:border-l sm:border-t-0">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Router pick strictly better</p>
              <p className="text-3xl font-semibold text-foreground">{pct(results.overall_shadow_win_rate_pct)}</p>
            </div>
          </div>
          <TierResultsTable groups={results.groups} />
        </>
      ) : (
        <p className="px-6 py-8 text-center text-sm text-muted-foreground">
          {active
            ? "Collecting verdicts — results appear as sampled requests are judged."
            : "No verdicts were recorded for this job."}
        </p>
      )}
    </Card>
  );
};

/** Kept in sync with DEFAULT_SHADOW_EVAL_JUDGE_MODEL on the backend. */
const DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5";

const StartForm: React.FC<{ accessToken: string | null }> = ({ accessToken }) => {
  const [apiKeyId, setApiKeyId] = useState("");
  const [routerName, setRouterName] = useState("");
  const [percentage, setPercentage] = useState("10");
  const [judgeModel, setJudgeModel] = useState(DEFAULT_JUDGE_MODEL);
  const start = useStartShadowEval();

  const parsedPct = Number.parseFloat(percentage);
  const valid =
    Boolean(accessToken) && apiKeyId.trim() !== "" && routerName.trim() !== "" && parsedPct > 0 && parsedPct <= 100;

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-foreground">Start a shadow eval</CardTitle>
        <p className="text-xs text-muted-foreground">
          Duplicates a sampled slice of the key&apos;s traffic through the auto-router and has an LLM judge compare both
          answers blind. The router&apos;s answers are never served to users. Judge calls bill to the proxy — an
          estimate is shown before anything runs.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-3">
          <Input
            placeholder="Key hash (token) to shadow"
            value={apiKeyId}
            onChange={(e) => setApiKeyId(e.target.value)}
          />
          <Input
            placeholder="Auto-router name (e.g. claude-auto)"
            value={routerName}
            onChange={(e) => setRouterName(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <Input
              type="number"
              min={0.1}
              max={100}
              step={0.1}
              className="w-24"
              value={percentage}
              onChange={(e) => setPercentage(e.target.value)}
            />
            <span className="text-sm text-muted-foreground">% of traffic</span>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <Input
            placeholder={`Judge model (default: ${DEFAULT_JUDGE_MODEL})`}
            value={judgeModel}
            onChange={(e) => setJudgeModel(e.target.value)}
          />
        </div>
        {start.error ? (
          <p className="text-sm text-destructive">
            {start.error instanceof ApiError ? start.error.message : "Failed to start shadow eval"}
          </p>
        ) : null}
        <Button
          disabled={!valid || start.isPending}
          onClick={() =>
            start.mutate({
              body: {
                api_key_id: apiKeyId.trim(),
                router_name: routerName.trim(),
                shadow_percentage: parsedPct,
                judge_model: judgeModel.trim() || DEFAULT_JUDGE_MODEL,
              },
            })
          }
        >
          {start.isPending ? "Starting…" : "Start shadow eval"}
        </Button>
      </CardContent>
    </Card>
  );
};

interface ShadowEvalSectionProps {
  accessToken: string | null;
}

const ShadowEvalSection: React.FC<ShadowEvalSectionProps> = ({ accessToken }) => {
  const { data: jobs, error } = useShadowEvalJobs(accessToken);
  const stop = useStopShadowEval();

  // Most recent job carries the section; older jobs list below it.
  const latest = useMemo(() => jobs?.[0] ?? null, [jobs]);
  const { data: latestDetail } = useShadowEvalJob(accessToken, latest?.job_id ?? null);

  if (error instanceof ApiError && error.status === 403) return null; // admin-only section

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-2">
        <h3 className="text-lg font-semibold text-foreground">Shadow eval</h3>
        <p className="text-xs text-muted-foreground">
          pre-adoption quality check: your current model vs. what the router would have picked
        </p>
      </div>

      {latest && latestDetail ? (
        <JobResults
          job={latestDetail}
          onStop={() => stop.mutate({ params: { path: { job_id: latestDetail.job_id } } })}
          stopPending={stop.isPending}
        />
      ) : null}

      {!latest || (latestDetail && latestDetail.status !== "pending" && latestDetail.status !== "running") ? (
        <StartForm accessToken={accessToken} />
      ) : null}
    </div>
  );
};

export default ShadowEvalSection;
