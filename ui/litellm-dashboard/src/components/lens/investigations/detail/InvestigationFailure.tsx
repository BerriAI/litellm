import type { ComponentProps } from "react";
import { runTime } from "../../model/format";
import { type Job } from "../../model/types";
import { cn } from "@/lib/cva.config";

export type InvestigationFailureProps = ComponentProps<"div"> & {
  job: Job;
  connected: boolean;
};

export function InvestigationFailure({ job, connected, className, ...props }: InvestigationFailureProps) {
  return (
    <div
      {...props}
      data-slot="investigation-failure"
      role="alert"
      className={cn("space-y-2 rounded-md border border-destructive/20 p-3 text-sm", className)}
    >
      <p className="font-medium text-destructive">This investigation did not finish</p>
      <p>{job.error}</p>
      <details>
        <summary className="cursor-pointer text-xs text-muted-foreground">Troubleshooting details</summary>
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
        <p className="mt-2 text-xs text-muted-foreground">
          Use the run ID to find the error in proxy and worker logs. Check the worker key&apos;s model permissions and
          budget before retrying.
        </p>
      </details>
    </div>
  );
}
