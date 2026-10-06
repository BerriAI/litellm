import type { ComponentProps } from "react";
import { runTime } from "../../model/format";
import { type Job } from "../../model/types";
import { failedTaskSummary, isPartial } from "../../model/status";
import { cn } from "@/lib/cva.config";

export type InvestigationFailureProps = ComponentProps<"div"> & {
  job: Job;
  connected: boolean;
};

export function InvestigationFailure({ job, connected, className, ...props }: InvestigationFailureProps) {
  const partial = isPartial(job);
  return (
    <div
      {...props}
      data-slot="investigation-failure"
      role={partial ? "status" : "alert"}
      className={cn(
        "space-y-2 rounded-md border p-3 text-sm",
        partial ? "border-warning/30" : "border-destructive/20",
        className,
      )}
    >
      <p className={cn("font-medium", !partial && "text-destructive")}>
        {partial ? "Partial results" : "This investigation did not finish"}
      </p>
      {partial && <p>{failedTaskSummary(job)}. Valid results are preserved.</p>}
      <details open={!partial}>
        <summary className="cursor-pointer text-xs text-muted-foreground">Run details</summary>
        <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-xs" aria-label="Investigation error">
          {job.error}
        </pre>
        <dl className="mt-2 space-y-1 text-xs text-muted-foreground">
          <div>
            <dt className="inline">Run: </dt>
            <dd className="inline">{job.id}</dd>
          </div>
          <div>
            <dt className="inline">Model: </dt>
            <dd className="inline">{job.settings.model}</dd>
          </div>
          <div>
            <dt className="inline">Worker: </dt>
            <dd className="inline">{connected ? "Connected now" : "Not connected"}</dd>
          </div>
          <div>
            <dt className="inline">Started: </dt>
            <dd className="inline">{runTime(job.created_at)}</dd>
          </div>
        </dl>
      </details>
    </div>
  );
}
