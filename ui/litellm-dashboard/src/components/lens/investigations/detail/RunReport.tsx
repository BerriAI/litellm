"use client";

import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useNow } from "@/hooks/useNow";

import { analysisElapsed } from "../../model/progress";
import { money } from "../../model/format";
import { modelsUsed, shortTime, windowLabel } from "../../model/inbox";
import type { Finding, Job } from "../../model/types";
import { InvestigationProgress } from "../InvestigationProgress";
import { StepFeed } from "../StepFeed";

export interface RunReportProps {
  readonly job: Job | undefined;
  /** Null when the run predates saved result snapshots. */
  readonly findings: readonly Finding[] | null | undefined;
  readonly connected: boolean;
  readonly picker: ReactNode;
  readonly onCancel?: () => void;
}

const STATUS_LABEL: Record<Job["status"], string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

const plural = (count: number, noun: string) => `${count.toLocaleString()} ${noun}${count === 1 ? "" : "s"}`;
const isActive = (job: Job) => job.status === "queued" || job.status === "running";

function headline(job: Job, findings: readonly Finding[] | null | undefined, connected: boolean): string {
  const runs = plural(job.coverage?.screened ?? 0, "run");
  const issues = findings?.filter((f) => f.kind === "issue").length ?? 0;
  const patterns = findings?.filter((f) => f.kind === "pattern").length ?? 0;
  switch (job.status) {
    case "queued":
      return connected ? "Waiting for the analyzer to pick this up" : "Waiting for an analyzer to connect";
    case "running":
      return "Investigating now";
    case "failed":
      return "Stopped before it finished";
    case "cancelled":
      return `Cancelled after reviewing ${runs}`;
    case "completed":
      if (!findings) return `Reviewed ${runs}`;
      if (issues + patterns === 0) return `Nothing found across ${runs}`;
      return `Found ${[issues && plural(issues, "issue"), patterns && plural(patterns, "pattern")].filter(Boolean).join(" and ")} across ${runs}`;
  }
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-1 truncate text-lg font-semibold tabular-nums">{value}</dd>
      {note && <dd className="truncate text-xs text-muted-foreground">{note}</dd>}
    </div>
  );
}

function coverageNote(job: Job): string | undefined {
  const { partial = 0, unassessable = 0, inconclusive = 0 } = job.coverage ?? {};
  const notes = [
    partial && `${partial} partial`,
    unassessable && `${unassessable} unreadable`,
    inconclusive && `${inconclusive} inconclusive`,
  ].filter(Boolean);
  return notes.length ? notes.join(" · ") : undefined;
}

function issueNote(job: Job, high: number): string | undefined {
  if (isActive(job)) return "When the run finishes";
  return high ? `${high} high priority` : undefined;
}

function RunStats({ job, findings, now }: { job: Job; findings: readonly Finding[] | null | undefined; now: number }) {
  const active = isActive(job);
  const known = job.status === "completed" && findings != null;
  const calls = (job.steps ?? []).filter((step) => step.kind === "model").length;
  const { screened = 0, selected = 0 } = job.coverage ?? {};
  const issues = findings?.filter((f) => f.kind === "issue") ?? [];
  const high = issues.filter((f) => f.priority === "high").length;
  const end = job.finished_at ? Date.parse(job.finished_at) : now;
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
      <Stat label="Cost" value={money(job.cost ?? 0)} note={plural(calls, "model call")} />
      <Stat
        label="Duration"
        value={active || job.finished_at ? analysisElapsed(job.created_at, end) : "–"}
        note={`Started ${shortTime(job.created_at)}`}
      />
      <Stat
        label="Runs reviewed"
        value={`${screened.toLocaleString()} / ${selected.toLocaleString()}`}
        note={coverageNote(job)}
      />
      <Stat label="Issues" value={known ? issues.length.toLocaleString() : "–"} note={issueNote(job, high)} />
    </dl>
  );
}

function RunFailure({ job, connected }: { job: Job; connected: boolean }) {
  return (
    <div role="alert" className="space-y-2 rounded-md bg-destructive/5 p-3">
      <pre
        aria-label="Investigation error"
        className="whitespace-pre-wrap break-words font-mono text-xs text-destructive"
      >
        {job.error}
      </pre>
      <p className="flex flex-wrap gap-x-3 text-xs text-muted-foreground">
        <span>
          Run <span className="font-mono">{job.id}</span>
        </span>
        <span>
          Model <span>{job.settings.model}</span>
        </span>
        <span>Worker {connected ? "connected now" : "not connected"}</span>
      </p>
    </div>
  );
}

function RunLog({ job }: { job: Job }) {
  const models = modelsUsed(job.steps ?? []);
  return (
    <Collapsible className="border-t pt-3">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <CollapsibleTrigger className="group flex items-center gap-1 font-medium text-foreground hover:underline">
          <ChevronRight className="size-3.5 transition-transform group-data-[panel-open]:rotate-90" />
          Activity log
        </CollapsibleTrigger>
        <span className="truncate">
          {job.trigger === "manual" ? "Manual" : "Scheduled"} · {windowLabel(job)}
          {models.length > 0 && ` · ${models.join(", ")}`}
        </span>
      </div>
      <CollapsibleContent className="pt-3">
        <StepFeed job={job} />
      </CollapsibleContent>
    </Collapsible>
  );
}

export function RunReport({ job, findings, connected, picker, onCancel }: RunReportProps) {
  const active = job !== undefined && isActive(job);
  const now = useNow(active ? 1000 : 60000);
  if (!job) {
    return (
      <section aria-label="Run report" className="flex items-center justify-between gap-3 rounded-lg border p-4">
        <p className="text-sm text-muted-foreground">No runs yet. Run it now to get the first report.</p>
        {picker}
      </section>
    );
  }
  return (
    <section aria-label="Run report" className="space-y-4 rounded-lg border p-4">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p
            data-state={job.status}
            className="text-xs font-medium text-muted-foreground data-[state=completed]:text-success data-[state=failed]:text-destructive data-[state=running]:text-info"
          >
            {STATUS_LABEL[job.status]}
          </p>
          <h3 className="mt-0.5 text-base font-semibold">{headline(job, findings, connected)}</h3>
        </div>
        {picker}
      </header>
      {active && <InvestigationProgress key={job.id} job={job} now={now} onCancel={onCancel} />}
      {job.error && <RunFailure job={job} connected={connected} />}
      <RunStats job={job} findings={findings} now={now} />
      <RunLog job={job} />
    </section>
  );
}
