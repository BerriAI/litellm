import { useState } from "react";
import { ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RunList } from "./ActivityScope";
import type { Job } from "./engineData";

function assessmentLabel(assessment: Job["assessments"][number] | undefined): string {
  if (!assessment) return "Not reviewed";
  if (assessment.cannot_assess) return "Insufficient evidence";
  return assessment.issue_checks?.length ? "Issue observed" : "No issue observed";
}

export function LensRuns({ job, onOpen }: { job?: Job; onOpen: (id: string) => void }) {
  const [runOffset, setRunOffset] = useState(0);
  const [runFilter, setRunFilter] = useState("all");
  const assessments = new Map(job?.assessments?.map((a) => [a.execution_id, a]));
  const visibleRuns = (job?.sample?.executions ?? []).filter((run) => {
    const assessment = assessments.get(run.id);
    if (runFilter === "all") return true;
    if (runFilter === "unknown") return !assessment || assessment.cannot_assess;
    if (runFilter === "clear") return assessment && !assessment.cannot_assess && !assessment.issue_checks?.length;
    return assessment?.issue_checks?.includes(runFilter);
  });
  return (
    <>
      <p className="text-sm font-medium">Runs in the selected batch</p>
      <p className="text-xs text-muted-foreground">
        {job?.sample?.executions.length ?? 0} selected from {job?.sample?.eligible ?? 0} matches. Open a run to inspect
        its original activity.
      </p>
      <label className="grid gap-2 text-sm">
        Review outcome
        <select
          aria-label="Filter reviewed runs"
          className="rounded-md border bg-background p-2"
          value={runFilter}
          onChange={(e) => {
            setRunFilter(e.target.value);
            setRunOffset(0);
          }}
        >
          <option value="all">All selected runs</option>
          <option value="clear">No issue observed</option>
          <option value="unknown">Insufficient evidence or not reviewed</option>
          {job?.settings?.context && <option value="expected_behavior">Expected behavior deviation</option>}
          {job?.settings?.checks.map((check) => (
            <option key={check.id} value={check.id}>
              {check.instruction}
            </option>
          ))}
        </select>
      </label>
      <p className="text-xs text-muted-foreground">
        These are per-run observations. Findings above investigate and group them with original evidence.
      </p>
      <div className="max-h-[480px] overflow-y-auto rounded-lg border px-4 divide-y">
        {visibleRuns.slice(runOffset, runOffset + 50).map((run) => (
          <div key={run.id} className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <RunList executions={[run]} />
              <p className="pb-3 text-xs text-muted-foreground">{assessmentLabel(assessments.get(run.id))}</p>
            </div>
            <Button size="sm" variant="ghost" onClick={() => onOpen(run.id)}>
              Open {run.source === "traces" ? "run" : "request"}
              <ArrowUpRight className="size-3" />
            </Button>
          </div>
        ))}
        {!job?.sample?.executions.length && (
          <p className="py-4 text-sm text-muted-foreground">
            The selected runs appear here when an analyzer starts the scan.
          </p>
        )}
      </div>
      <div className="flex items-center justify-between text-sm">
        <Button variant="ghost" disabled={!runOffset} onClick={() => setRunOffset(Math.max(0, runOffset - 50))}>
          Previous runs
        </Button>
        <span>{visibleRuns.length} matching runs</span>
        <Button
          variant="ghost"
          disabled={runOffset + 50 >= visibleRuns.length}
          onClick={() => setRunOffset(runOffset + 50)}
        >
          Next runs
        </Button>
      </div>
    </>
  );
}
