"use client";

import React, { useMemo, useState } from "react";

import { useInfiniteKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useAutoRouters, usePlainModelGroups } from "@/app/(dashboard)/hooks/models/useModels";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError } from "@/lib/http/client";

import { usd } from "./costOptimizationUtils";
import {
  useShadowEvalJob,
  useShadowEvalJobs,
  useStartShadowEval,
  useStopShadowEval,
  type ShadowEvalJob,
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

const routerMatchedOrBeatPct = (
  direction: ShadowEvalDirection,
  results: NonNullable<ShadowEvalJob["results"]>,
): number =>
  direction === "reverse"
    ? 100 - results.overall_shadow_win_rate_pct
    : results.overall_shadow_win_rate_pct + results.overall_tie_rate_pct;

const jobHeadline = (job: ShadowEvalJob): React.ReactNode =>
  job.direction === "reverse" ? (
    <>
      Comparing <span className="font-mono text-xs">{job.router_name}</span> to{" "}
      <span className="font-mono text-xs">{job.baseline_model}</span> on {job.shadow_percentage}% of its traffic
    </>
  ) : (
    <>
      Shadowing {job.shadow_percentage}% via <span className="font-mono text-xs">{job.router_name}</span>
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
  running: "bg-blue-50 text-blue-700",
  completed: "bg-emerald-50 text-emerald-700",
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
        {["Judged turns", "Router wins", `${otherArmLabel(direction)} wins`, "Ties", "Judge confidence"].map(
          (label) => (
            <TableHead key={label} className="text-right">
              {label}
            </TableHead>
          ),
        )}
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
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

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
    { label: "Router won", value: routerWins, fill: "bg-emerald-500" },
    { label: "Tie", value: ties, fill: "bg-emerald-200" },
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

const emptyResultsText = (job: ShadowEvalJob, resultsError: boolean): string => {
  if (resultsError) return "Results could not be loaded. Retrying.";
  if (isActive(job)) return "Collecting verdicts. Results appear as sampled requests are judged.";
  if (job.judged_count === 0) return "No verdicts were recorded for this job.";
  return "Loading results...";
};

const ResultsBody: React.FC<{ job: ShadowEvalJob; resultsError?: boolean }> = ({ job, resultsError = false }) => {
  const results = job.results;
  if (!results || (results.by_tier.length === 0 && results.by_current_model.length === 0)) {
    return <p className="px-6 py-8 text-center text-sm text-muted-foreground">{emptyResultsText(job, resultsError)}</p>;
  }
  return (
    <>
      <div className="flex flex-col gap-1 border-b px-6 py-4">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Router matched or beat {job.direction === "reverse" ? "the baseline" : "your current model"}
        </p>
        <p className="text-3xl font-semibold text-foreground">{pct(routerMatchedOrBeatPct(job.direction, results))}</p>
        <p className="text-xs text-muted-foreground">of {(job.judged_count ?? 0).toLocaleString()} judged responses</p>
      </div>
      <VerdictBar direction={job.direction} results={results} />
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
              {(job.judged_count ?? 0).toLocaleString()} of {job.max_turns.toLocaleString()} turns judged ·{" "}
              {(job.error_count ?? 0).toLocaleString()} errored · {usd(job.judge_spend ?? 0)} judge spend
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
        <p className="border-b bg-red-50 px-6 py-2 text-xs text-destructive">
          Last failure: <span className="font-mono">{job.last_error}</span>
        </p>
      )}
      <ResultsBody job={job} resultsError={resultsError} />
    </Card>
  );
};

const RECOMMENDED_JUDGE_MODELS = ["anthropic/claude-sonnet-5", "openai/gpt-4o", "gemini/gemini-2.5-pro"] as const;

interface CostMapEntry {
  litellm_provider?: string;
  mode?: string;
}

const useChatModelNames = (): string[] => {
  const { data: costMap } = useModelCostMap();
  return useMemo(() => {
    if (!costMap) return [];
    const chatModels = Object.entries(costMap as Record<string, CostMapEntry>)
      .filter(([, value]) => value?.mode === "chat" && value?.litellm_provider)
      .map(([key, value]) => (key.startsWith(`${value.litellm_provider}/`) ? key : `${value.litellm_provider}/${key}`));
    return [...new Set(chatModels)].toSorted((a, b) => a.localeCompare(b));
  }, [costMap]);
};

const useJudgeModelOptions = (): SearchSelectOption[] => {
  const chatModels = useChatModelNames();
  return useMemo(() => {
    const pinned: SearchSelectOption[] = RECOMMENDED_JUDGE_MODELS.map((model) => ({
      label: model,
      value: model,
      sublabel: "Recommended",
    }));
    const pinnedNames = new Set<string>(RECOMMENDED_JUDGE_MODELS);
    const rest = chatModels.filter((model) => !pinnedNames.has(model)).map((model) => ({ label: model, value: model }));
    return [...pinned, ...rest];
  }, [chatModels]);
};

const useBaselineModelOptions = (): SearchSelectOption[] => {
  const configuredGroups = usePlainModelGroups();
  const chatModels = useChatModelNames();
  return useMemo(() => {
    const configured = [...configuredGroups]
      .toSorted((a, b) => a.localeCompare(b))
      .map((model) => ({ label: model, value: model, sublabel: "Configured on this gateway" }));
    const rest = chatModels
      .filter((model) => !configuredGroups.has(model))
      .map((model) => ({ label: model, value: model }));
    return [...configured, ...rest];
  }, [configuredGroups, chatModels]);
};

const DIRECTION_OPTIONS: readonly { value: ShadowEvalDirection; label: string }[] = [
  { value: "forward", label: "Adoption check: key's traffic vs the router" },
  { value: "reverse", label: "Regression check: router's picks vs a baseline" },
] as const;

const START_FORM_DESCRIPTION: Record<ShadowEvalDirection, string> = {
  forward:
    "Duplicates a sampled slice of the key's traffic through the auto-router and has an LLM judge compare both answers blind. The router's answers are never served to users; judge calls bill to the shadowed key.",
  reverse:
    "Duplicates a sampled slice of the traffic the auto-router already serves against a fixed baseline model and has an LLM judge compare both answers blind. The baseline's answers are never served to users; judge calls bill to the shadowed key.",
};

const DURATION_OPTIONS = [
  { value: "1", label: "1 day" },
  { value: "3", label: "3 days" },
  { value: "7", label: "7 days" },
  { value: "14", label: "14 days" },
  { value: "30", label: "30 days" },
] as const;

const Field: React.FC<{ label: string; htmlFor?: string; className?: string; children: React.ReactNode }> = ({
  label,
  htmlFor,
  className,
  children,
}) => (
  <div className={`space-y-1.5 ${className ?? ""}`}>
    <Label htmlFor={htmlFor} className="text-xs">
      {label}
    </Label>
    {children}
  </div>
);

const KeySelect: React.FC<{ value: string; onChange: (token: string) => void }> = ({ value, onChange }) => {
  const [search, setSearch] = useState("");
  const { data, isPending, isError, fetchNextPage, hasNextPage, isFetchingNextPage } = useInfiniteKeys(50, {
    selectedKeyAlias: search || null,
  });
  const options = useMemo<SearchSelectOption[]>(
    () =>
      (data?.pages ?? [])
        .flatMap((page) => page.keys)
        .map((key) => ({
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
      onLoadMore={() => void fetchNextPage()}
      hasNextPage={hasNextPage}
      isFetchingNextPage={isFetchingNextPage}
      isLoading={isPending}
      placeholder="Search keys by alias"
      emptyText="No matching keys"
      errorText={isError ? "Keys could not be loaded. Refresh the page to retry." : undefined}
    />
  );
};

const StartForm: React.FC = () => {
  const { accessToken } = useAuthorized();
  const [apiKeyId, setApiKeyId] = useState("");
  const [routerName, setRouterName] = useState("");
  const [direction, setDirection] = useState<ShadowEvalDirection>("forward");
  const [baselineModel, setBaselineModel] = useState("");
  const [percentage, setPercentage] = useState("10");
  const [durationDays, setDurationDays] = useState("7");
  const [judgeModel, setJudgeModel] = useState("");
  const [maxTurns, setMaxTurns] = useState("200");
  const { data: autoRouters } = useAutoRouters();
  const judgeModelOptions = useJudgeModelOptions();
  const baselineModelOptions = useBaselineModelOptions();
  const start = useStartShadowEval();

  const routerOptions = useMemo<SearchSelectOption[]>(() => {
    const names = new Set(
      (autoRouters ?? []).map((deployment) => deployment.model_name).filter((name): name is string => Boolean(name)),
    );
    return [...names].toSorted().map((name) => ({ label: name, value: name }));
  }, [autoRouters]);

  const parsedPct = Number.parseFloat(percentage);
  const percentageValid = parsedPct >= 0.1 && parsedPct <= 100;
  const parsedMaxTurns = Number.parseInt(maxTurns, 10);
  const maxTurnsValid = parsedMaxTurns >= 1 && parsedMaxTurns <= 2000;
  const filled =
    [apiKeyId, routerName, judgeModel].every((field) => field !== "") &&
    (direction === "forward" || baselineModel !== "");
  const boundsValid = percentageValid && maxTurnsValid;
  const valid = Boolean(accessToken) && filled && boundsValid;
  const handleStart = () => {
    const startBody = {
      api_key_id: apiKeyId,
      router_name: routerName,
      direction,
      ...(direction === "reverse" ? { baseline_model: baselineModel } : {}),
      shadow_percentage: parsedPct,
      duration_days: Number.parseInt(durationDays, 10),
      max_turns: parsedMaxTurns,
      judge_model: judgeModel,
    };
    start.mutate(startBody);
  };

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-foreground">Start a shadow eval</CardTitle>
        <p className="text-xs text-muted-foreground">{START_FORM_DESCRIPTION[direction]}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-3">
          <Field label="Direction">
            <Select
              value={direction}
              onValueChange={(v: string | null) => setDirection(v === "reverse" ? "reverse" : "forward")}
            >
              <SelectTrigger className="w-full">
                <SelectValue>{DIRECTION_OPTIONS.find((o) => o.value === direction)?.label}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {DIRECTION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label="Key to shadow" htmlFor="shadow-eval-key">
            <KeySelect value={apiKeyId} onChange={setApiKeyId} />
          </Field>
          <Field label="Auto-router">
            <SearchSelect
              options={routerOptions}
              value={routerName}
              onValueChange={setRouterName}
              placeholder="Select an auto-router"
              emptyText="No auto-routers configured"
            />
          </Field>
          <Field label="Traffic sampled" htmlFor="shadow-eval-pct">
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
            <div>
              {percentage.trim() !== "" && !percentageValid && (
                <p className="text-xs text-destructive">Enter a value from 0.1 to 100</p>
              )}
            </div>
          </Field>
          <Field label="Duration">
            <Select value={durationDays} onValueChange={(v: string | null) => setDurationDays(v ?? "7")}>
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
          </Field>
          <Field label="Turn budget">
            <div className="flex items-center gap-2">
              <Input
                type="number"
                min={1}
                max={2000}
                className="w-24"
                value={maxTurns}
                onChange={(e) => setMaxTurns(e.target.value)}
              />
              <span className="text-sm text-muted-foreground">turns judged, max</span>
            </div>
            {maxTurns.trim() !== "" && !maxTurnsValid && (
              <p className="text-xs text-destructive">Enter a value from 1 to 2000</p>
            )}
          </Field>
          {direction === "reverse" && (
            <Field label="Baseline model">
              <SearchSelect
                options={baselineModelOptions}
                value={baselineModel}
                onValueChange={setBaselineModel}
                placeholder="Select a baseline model"
                emptyText="No chat models available"
              />
            </Field>
          )}
          <Field label="Judge model" className="sm:col-span-2">
            <SearchSelect
              options={judgeModelOptions}
              value={judgeModel}
              onValueChange={setJudgeModel}
              placeholder="Select a judge model"
              emptyText="No chat models available"
            />
          </Field>
        </div>
        <Button disabled={!valid || start.isPending} onClick={handleStart}>
          {start.isPending ? "Starting..." : "Start shadow eval"}
        </Button>
      </CardContent>
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
                `${shown.judged_count.toLocaleString()} judged · ${(shown.error_count ?? 0).toLocaleString()} errored · ${usd(shown.judge_spend ?? 0)} judge spend · `}
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
          Blind-judge the auto-router on your real traffic: against the models a key uses today before switching, or
          against a fixed baseline after it has switched.
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
