import { useState } from "react";
import { ChevronRight, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { NextCheck } from "./LensProgress";
import { DurationInput } from "./DurationInput";
import { lensStatus, runTime, scopeLabel, type Lens, type Settings, type Job } from "./lensData";

export function InvestigationList({
  lenses,
  connected,
  onSelect,
}: {
  lenses: Lens[];
  connected: boolean;
  onSelect: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const shown = lenses.filter((lens) =>
    `${lens.settings.name} ${scopeLabel(lens.settings)}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <section aria-label="Saved investigations" className="space-y-4">
      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" />
        <Input
          aria-label="Search investigations"
          placeholder="Search investigations"
          className="pl-9"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
      <div className="divide-y border-y">
        {shown.map((lens) => (
          <button
            key={lens.id}
            onClick={() => onSelect(lens.id)}
            className="group grid w-full grid-cols-[minmax(0,1fr)_16px] items-center gap-x-4 gap-y-2 py-4 text-left hover:bg-muted/30 focus-visible:outline-2 focus-visible:outline-ring sm:grid-cols-[minmax(0,1fr)_auto_16px]"
          >
            <div className="col-start-1 row-start-1 min-w-0">
              <p className="text-sm font-medium">{lens.settings.name}</p>
              <p className="mt-1 truncate text-xs text-muted-foreground">{scopeLabel(lens.settings)}</p>
            </div>
            <div className="col-start-1 row-start-2 text-xs sm:col-start-2 sm:row-start-1 sm:text-right">
              <p className={lens.jobs[0]?.status === "failed" ? "text-destructive" : "text-muted-foreground"}>
                {lensStatus(lens, connected)}
              </p>
              <p className="mt-1 text-muted-foreground">
                {lens.jobs[0] ? runTime(lens.jobs[0].created_at) : "Not run yet"}
              </p>
            </div>
            <ChevronRight className="col-start-2 row-start-1 size-4 text-muted-foreground sm:col-start-3" />
          </button>
        ))}
        {!shown.length && <p className="py-8 text-sm text-muted-foreground">No investigations match your search.</p>}
      </div>
    </section>
  );
}

export function MonitoringSetup({
  settings,
  ready,
  onSave,
  onClose,
}: {
  settings: Settings;
  ready: boolean;
  onSave: (settings: Settings) => Promise<void>;
  onClose: () => void;
}) {
  const [interval, setInterval] = useState(settings.interval_minutes ?? 30);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const save = async () => {
    setBusy(true);
    try {
      await onSave({ ...settings, enabled: true, interval_minutes: interval });
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not enable monitoring");
    } finally {
      setBusy(false);
    }
  };
  const validInterval = Number.isInteger(interval) && interval >= 1 && interval <= 10080;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-xl">Keep monitoring</DialogTitle>
          <DialogDescription>
            Repeat this investigation with the saved scope, sample, model, and budget.
          </DialogDescription>
        </DialogHeader>
        <DurationInput label="Check every" value={interval} onChange={setInterval} base="minutes" max={10080} />
        <p className="text-xs leading-5 text-muted-foreground">
          Each investigation looks back over the saved time range. The interval starts after the previous run finishes.
        </p>
        {!ready && (
          <p role="status" className="text-sm text-amber-700">
            Reconnect the worker before enabling monitoring.
          </p>
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button disabled={busy || !ready || !validInterval} onClick={() => void save()}>
            {busy ? "Saving…" : "Enable monitoring"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

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

export function InvestigationFailure({ job, connected }: { job: Job; connected: boolean }) {
  return (
    <div role="alert" className="space-y-2 rounded-md border border-destructive/20 p-3 text-sm">
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
