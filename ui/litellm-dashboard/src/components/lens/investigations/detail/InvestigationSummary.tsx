import type { ComponentProps } from "react";
import { NextCheck } from "./JobMeta";
import { lensStatus } from "../../model/status";
import { runTime } from "../../model/format";
import { type Lens } from "../../model/types";
import { cn } from "@/lib/cva.config";

export type InvestigationSummaryProps = ComponentProps<"div"> & {
  lens: Lens;
  connected: boolean;
};

export function InvestigationSummary({ lens, connected, className, ...props }: InvestigationSummaryProps) {
  const lastCompleted = lens.jobs.find((job) => job.status === "completed");
  const lastSuccess = lastCompleted?.finished_at ?? lens.last_scan_at;
  const spent = lens.budget_month === new Date().toISOString().slice(0, 7) ? lens.spent ?? 0 : 0;
  return (
    <div
      {...props}
      data-slot="investigation-summary"
      className={cn("flex flex-wrap gap-x-8 gap-y-3 border-y py-3 text-xs text-muted-foreground", className)}
    >
      <span>
        Latest run:{" "}
        <strong
          data-state={lens.jobs[0]?.status === "failed" ? "failed" : "ok"}
          className="font-medium data-[state=failed]:text-destructive data-[state=ok]:text-foreground"
        >
          {lensStatus(lens, connected)}
        </strong>
      </span>
      <span>
        Last success: <span className="text-foreground">{lastSuccess ? runTime(lastSuccess) : "Not yet"}</span>
      </span>
      <span>
        This month:{" "}
        <span className="text-foreground">
          ${spent.toFixed(3)} / ${lens.settings.monthly_budget ?? 100}
        </span>
      </span>
      {lens.settings.enabled && (
        <span>
          Monitoring every {lens.settings.interval_minutes} minutes
          <NextCheck lens={lens} />
        </span>
      )}
    </div>
  );
}
