import { NextCheck } from "../InvestigationProgress";
import { lensStatus } from "../../model/status";
import { runTime } from "../../model/format";
import { type Lens } from "../../model/types";

export function InvestigationSummary({ lens, connected }: { lens: Lens; connected: boolean }) {
  const lastCompleted = lens.jobs.find((job) => job.status === "completed");
  const lastSuccess = lastCompleted?.finished_at ?? lens.last_scan_at;
  const spent = lens.budget_month === new Date().toISOString().slice(0, 7) ? lens.spent ?? 0 : 0;
  return (
    <div className="flex flex-wrap gap-x-8 gap-y-3 border-y py-3 text-xs text-muted-foreground">
      <span>
        Latest run:{" "}
        <strong className={`font-medium ${lens.jobs[0]?.status === "failed" ? "text-destructive" : "text-foreground"}`}>
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
