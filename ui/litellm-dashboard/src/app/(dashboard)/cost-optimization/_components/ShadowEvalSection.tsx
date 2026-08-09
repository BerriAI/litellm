"use client";

import React, { useMemo, useState } from "react";

import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useAutoRouters } from "@/app/(dashboard)/hooks/models/useModels";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
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

const endsIn = (endsAt: string | null | undefined): string | null => {
  if (!endsAt) return null;
  const remainingMs = new Date(endsAt).getTime() - Date.now();
  if (Number.isNaN(remainingMs)) return null;
  if (remainingMs <= 0) return "ending now";
  const days = Math.round(remainingMs / 86_400_000);
  if (days >= 2) return `ends in ${days} days`;
  const hours = Math.round(remainingMs / 3_600_000);
  return hours >= 2 ? `ends in ${hours} hours` : "ends within the hour";
};

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

const HOW_IT_WORKS_STEPS = [
  "A sampled slice of the key's live traffic is picked at random. Users are never affected — they keep getting answers from your current models.",
  "Each sampled request is quietly duplicated through the auto-router, which classifies the prompt and picks whatever model it would have picked.",
  "An LLM judge compares the two answers blind: it sees them only as “A” and “B” in random order, never which system produced which.",
  "Verdicts accumulate below, broken down by the router's difficulty tier. Shadow and judge calls bill to the shadowed key, capped per job.",
] as const;

const HowThisWorks: React.FC = () => {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="text-xs text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground">
        {open ? "Hide how this works" : "How this works"}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ol className="mt-2 max-w-3xl list-decimal space-y-1 pl-5 text-xs text-muted-foreground">
          {HOW_IT_WORKS_STEPS.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </CollapsibleContent>
    </Collapsible>
  );
};

const HeadTooltip: React.FC<{ label: string; tooltip: string; className?: string }> = ({
  label,
  tooltip,
  className,
}) => (
  <TableHead className={className}>
    <TooltipProvider delay={200}>
      <Tooltip>
        <TooltipTrigger render={<span />}>
          <span className="underline decoration-dotted underline-offset-2">{label}</span>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  </TableHead>
);

const LOW_SAMPLE_TOOLTIP = `Fewer than ${MIN_TURNS_FOR_CONFIDENCE} judged turns — treat as directional only.`;

const LowSampleFlag: React.FC = () => (
  <TooltipProvider delay={200}>
    <Tooltip>
      <TooltipTrigger render={<span className="ml-2 text-xs text-muted-foreground" />}>
        <span className="underline decoration-dotted underline-offset-2">(low sample)</span>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{LOW_SAMPLE_TOOLTIP}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const TierResultsTable: React.FC<{ groups: readonly ShadowEvalTierResult[] }> = ({ groups }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Router tier</TableHead>
        <TableHead className="text-right">Judged turns</TableHead>
        <TableHead className="text-right">Router wins</TableHead>
        <TableHead className="text-right">Current model wins</TableHead>
        <HeadTooltip
          className="text-right"
          label="Ties"
          tooltip="The judge saw the same quality from both. Ties count in the router's favor — same answer quality, usually at lower cost."
        />
        <HeadTooltip
          className="text-right"
          label="Judge confidence"
          tooltip="The judge's self-reported certainty in its verdicts (0 to 1), averaged over this tier's turns."
        />
      </TableRow>
    </TableHeader>
    <TableBody>
      {groups.map((g) => (
        <TableRow key={g.tier}>
          <TableCell className="font-medium text-foreground">
            {g.tier}
            {g.turn_count < MIN_TURNS_FOR_CONFIDENCE ? <LowSampleFlag /> : null}
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

const MetricLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <TooltipProvider delay={200}>
    <Tooltip>
      <TooltipTrigger render={<span className="w-fit text-[11px] uppercase tracking-wide text-muted-foreground" />}>
        <span className="underline decoration-dotted underline-offset-2">{label}</span>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const VerdictBar: React.FC<{ results: NonNullable<ShadowEvalJob["results"]> }> = ({ results }) => {
  const routerWins = results.overall_shadow_win_rate_pct;
  const ties = results.overall_tie_rate_pct;
  const currentWins = Math.max(0, 100 - routerWins - ties);
  const segments = [
    { label: "Router won", value: routerWins, className: "bg-emerald-500" },
    { label: "Tie", value: ties, className: "bg-emerald-200" },
    { label: "Current model won", value: currentWins, className: "bg-muted-foreground/30" },
  ];
  return (
    <div className="space-y-2 border-b px-6 py-4">
      <div className="flex h-2 w-full overflow-hidden rounded-full" role="img" aria-label="Verdict breakdown">
        {segments.map((segment) =>
          segment.value > 0 ? (
            <div
              key={segment.label}
              data-testid={`verdict-segment-${segment.label}`}
              className={segment.className}
              style={{ width: `${segment.value}%` }}
            />
          ) : null,
        )}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {segments.map((segment) => (
          <span key={segment.label} className="flex items-center gap-1.5">
            <span className={`size-2 rounded-full ${segment.className}`} />
            {segment.label} {pct(segment.value)}
          </span>
        ))}
      </div>
    </div>
  );
};

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
              {job.completed_count.toLocaleString()} responses judged · {job.failed_count.toLocaleString()} errored
              (shadow or judge call) ·{" "}
              {job.cost_actual != null ? `${usd(job.cost_actual)} judge spend` : "no judge spend yet"}
              {job.cost_estimate != null ? ` of ~${usd(job.cost_estimate)} estimated` : ""}
              {active && endsIn(job.ends_at) ? ` · ${endsIn(job.ends_at)}` : ""}
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
          <div className="flex flex-col justify-center gap-1 border-b px-6 py-4">
            <MetricLabel
              label="Router matched or beat your current model"
              tooltip="Router wins plus ties. A tie counts in the router's favor: the judge saw the same quality, and the router usually picked a cheaper model."
            />
            <p className="text-3xl font-semibold text-foreground">{okOrBetter != null ? pct(okOrBetter) : "—"}</p>
            <p className="text-xs text-muted-foreground">of {job.completed_count.toLocaleString()} judged responses</p>
          </div>
          <VerdictBar results={results} />
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

/** No default is sent — these are suggestions shown ahead of the rest of the catalog,
 * one per provider, so a pick doesn't depend on having one provider configured. */
const RECOMMENDED_JUDGE_MODELS = ["anthropic/claude-sonnet-5", "openai/gpt-4o", "gemini/gemini-2.5-pro"] as const;

interface CostMapEntry {
  litellm_provider?: string;
  mode?: string;
}

const useJudgeModelOptions = (): SearchSelectOption[] => {
  const { data: costMap } = useModelCostMap();
  return useMemo(() => {
    if (!costMap) return [];
    const seen = new Set<string>();
    const options: SearchSelectOption[] = [];
    for (const recommended of RECOMMENDED_JUDGE_MODELS) {
      seen.add(recommended);
      options.push({ label: recommended, value: recommended, sublabel: "Recommended" });
    }
    const rest: SearchSelectOption[] = [];
    for (const [key, value] of Object.entries(costMap as Record<string, CostMapEntry>)) {
      const provider = value?.litellm_provider;
      if (value?.mode !== "chat" || !provider) continue;
      const model = key.startsWith(`${provider}/`) ? key : `${provider}/${key}`;
      if (seen.has(model)) continue;
      seen.add(model);
      rest.push({ label: model, value: model });
    }
    rest.sort((a, b) => a.label.localeCompare(b.label));
    return [...options, ...rest];
  }, [costMap]);
};

const DURATION_OPTIONS = [
  { value: "1", label: "1 day" },
  { value: "3", label: "3 days" },
  { value: "7", label: "7 days" },
  { value: "14", label: "14 days" },
  { value: "30", label: "30 days" },
] as const;

const DEFAULT_DURATION_DAYS = "7";
const KEY_PAGE_SIZE = 50;

const KeySelect: React.FC<{ value: string; onChange: (token: string) => void }> = ({ value, onChange }) => {
  const [search, setSearch] = useState("");
  const { data, isPending } = useKeys(1, KEY_PAGE_SIZE, { selectedKeyAlias: search || null });
  const options = useMemo<SearchSelectOption[]>(
    () =>
      (data?.keys ?? []).map((key) => ({
        label: key.key_alias || key.key_name || key.token,
        value: key.token,
        sublabel: key.token,
      })),
    [data],
  );
  return (
    <PaginatedSearchSelect
      inputId="shadow-eval-key"
      options={options}
      value={value}
      onValueChange={onChange}
      onSearchChange={setSearch}
      onLoadMore={() => {}}
      isLoading={isPending}
      placeholder="Search keys by alias"
      emptyText="No matching keys"
    />
  );
};

const StartForm: React.FC<{ accessToken: string | null }> = ({ accessToken }) => {
  const [apiKeyId, setApiKeyId] = useState("");
  const [routerName, setRouterName] = useState("");
  const [percentage, setPercentage] = useState("10");
  const [durationDays, setDurationDays] = useState(DEFAULT_DURATION_DAYS);
  const [judgeModel, setJudgeModel] = useState("");
  const { data: autoRouters } = useAutoRouters();
  const judgeModelOptions = useJudgeModelOptions();
  const start = useStartShadowEval();

  const routerOptions = useMemo<SearchSelectOption[]>(() => {
    const names = new Set<string>();
    for (const deployment of autoRouters ?? []) {
      if (deployment.model_name) names.add(deployment.model_name);
    }
    return [...names].sort().map((name) => ({ label: name, value: name }));
  }, [autoRouters]);

  const parsedPct = Number.parseFloat(percentage);
  const percentageValid = parsedPct > 0 && parsedPct <= 100;
  const requiredFieldsPicked = apiKeyId.trim() !== "" && routerName.trim() !== "" && judgeModel.trim() !== "";
  const valid = Boolean(accessToken) && requiredFieldsPicked && percentageValid;

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-foreground">Start a shadow eval</CardTitle>
        <p className="text-xs text-muted-foreground">
          Duplicates a sampled slice of the key&apos;s traffic through the auto-router and has an LLM judge compare both
          answers blind. The router&apos;s answers are never served to users. Judge calls bill to the shadowed key — an
          estimate is shown before anything runs, and the job stops itself when the duration ends.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="space-y-1.5">
            <Label htmlFor="shadow-eval-key" className="text-xs">
              Key to shadow
            </Label>
            <KeySelect value={apiKeyId} onChange={setApiKeyId} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="shadow-eval-router" className="text-xs">
              Auto-router
            </Label>
            <SearchSelect
              options={routerOptions}
              value={routerName}
              onValueChange={setRouterName}
              placeholder="Select an auto-router"
              emptyText="No auto-routers configured"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="shadow-eval-pct" className="text-xs">
              Traffic sampled
            </Label>
            <div className="flex items-center gap-2">
              <Input
                id="shadow-eval-pct"
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
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="space-y-1.5">
            <Label className="text-xs">Duration</Label>
            <Select
              value={durationDays}
              onValueChange={(v: string | null) => setDurationDays(v ?? DEFAULT_DURATION_DAYS)}
            >
              <SelectTrigger className="w-full">
                <SelectValue>{DURATION_OPTIONS.find((o) => o.value === durationDays)?.label}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {DURATION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5 sm:col-span-2">
            <Label htmlFor="shadow-eval-judge" className="text-xs">
              Judge model
            </Label>
            <SearchSelect
              options={judgeModelOptions}
              value={judgeModel}
              onValueChange={setJudgeModel}
              placeholder="Select a judge model"
              emptyText="No chat models available"
            />
            <p className="text-xs text-muted-foreground">
              The judge only compares two answers blind — a mid-tier model is the sweet spot. Recommended:{" "}
              <span className="font-mono">anthropic/claude-sonnet-5</span>,{" "}
              <span className="font-mono">openai/gpt-4o</span>, or{" "}
              <span className="font-mono">gemini/gemini-2.5-pro</span>. Small &quot;nano/mini&quot; models give
              unreliable verdicts; frontier reasoning models cost more without changing outcomes.
            </p>
          </div>
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
                duration_days: Number.parseInt(durationDays, 10),
                judge_model: judgeModel,
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

const PreviousJob: React.FC<{
  accessToken: string | null;
  job: ShadowEvalJob;
}> = ({ accessToken, job }) => {
  const [expanded, setExpanded] = useState(false);
  const { data: detail } = useShadowEvalJob(accessToken, expanded ? job.job_id : null);
  const shown = detail ?? job;
  const results = shown.results;
  const okOrBetter = results ? results.overall_shadow_win_rate_pct + results.overall_tie_rate_pct : null;
  let summary = "no verdicts";
  if (okOrBetter != null) {
    summary = pct(okOrBetter);
  } else if (shown.completed_count > 0) {
    summary = "view results";
  }

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
            <p className="text-sm font-medium text-foreground">
              {shown.shadow_percentage}% via <span className="font-mono text-xs">{shown.router_name}</span>
            </p>
            <p className="text-xs text-muted-foreground">
              {shown.completed_count.toLocaleString()} judged · {shown.failed_count.toLocaleString()} failed ·{" "}
              {shown.cost_actual != null ? `${usd(shown.cost_actual)} judge spend` : "no judge spend yet"}
              {shown.created_at ? ` · ${new Date(shown.created_at).toLocaleDateString()}` : ""}
            </p>
          </div>
        </div>
        <span className="text-sm font-medium text-foreground">{summary}</span>
      </button>
      {expanded ? (
        <div className="px-6 pb-4">
          {results && results.groups.length > 0 ? (
            <TierResultsTable groups={results.groups} />
          ) : (
            <p className="text-xs text-muted-foreground">No verdicts were recorded for this evaluation.</p>
          )}
        </div>
      ) : null}
    </div>
  );
};

const PreviousJobs: React.FC<{
  accessToken: string | null;
  jobs: readonly ShadowEvalJob[];
}> = ({ accessToken, jobs }) => {
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
      {open ? (
        <div className="border-t">
          {jobs.map((job) => (
            <PreviousJob key={job.job_id} accessToken={accessToken} job={job} />
          ))}
        </div>
      ) : null}
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
  const previous = useMemo(() => jobs?.slice(1) ?? [], [jobs]);
  const { data: latestDetail } = useShadowEvalJob(accessToken, latest?.job_id ?? null);

  if (error instanceof ApiError && error.status === 403) return null; // admin-only section

  return (
    <div id="shadow-eval-section" className="space-y-4 scroll-mt-6">
      <div className="space-y-1">
        <div className="flex flex-wrap items-baseline gap-2">
          <h3 className="text-lg font-semibold text-foreground">Shadow eval</h3>
          <p className="text-sm text-muted-foreground">
            Would the auto-router have answered as well as the models you use today? Find out on your real traffic,
            before switching anything.
          </p>
        </div>
        <HowThisWorks />
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

      <PreviousJobs accessToken={accessToken} jobs={previous} />
    </div>
  );
};

export default ShadowEvalSection;
