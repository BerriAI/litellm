import Link from "next/link";
import { Check, Loader2 } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { LensPreviewButton } from "../LensPreviewButton";
import { uiHref } from "@/utils/uiHref";
import { cn } from "@/lib/cva.config";

export function InvestigationsWelcome({
  tracesReady,
  requestsReady = false,
  checking,
  traceError,
  connected,
  readOnly,
  onConnect,
  onCreate,
  onRetry,
  showPreview = false,
}: {
  tracesReady: boolean;
  requestsReady?: boolean;
  checking: boolean;
  traceError?: string;
  connected: boolean;
  readOnly: boolean;
  onConnect: () => void;
  onCreate: () => void;
  onRetry: () => void;
  showPreview?: boolean;
}) {
  const activityReady = tracesReady || requestsReady;
  const workerReady = activityReady && connected;
  const canCreate = workerReady && !readOnly;
  const canConnect = activityReady && !readOnly;
  const traceStatus = activityStatus(tracesReady, requestsReady, checking);
  const waitingForWorker = activityReady && !connected;
  const firstStepTitle = requestsReady && !tracesReady ? "Recorded activity" : "Set up traces";
  const traceButtonClass = buttonVariants({
    variant: activityReady ? "ghost" : "default",
    className: "col-start-2 w-fit sm:col-start-auto",
  });
  return (
    <section aria-labelledby="lens-welcome" className="max-w-3xl pb-6">
      {showPreview && <LensPreviewButton />}
      <h2 id="lens-welcome" className="text-xl font-semibold tracking-tight">
        Find what needs attention
      </h2>
      <p className="mt-2 max-w-lg text-sm leading-6 text-muted-foreground">
        Check how your agents behave. Get findings you can trace back to what happened.
      </p>
      <ol className="mt-8 divide-y border-y">
        <li className="grid grid-cols-[28px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 py-5 sm:grid-cols-[28px_minmax(0,1fr)_auto]">
          <Step number={1} complete={activityReady} checking={checking} active={!activityReady} />
          <div>
            <h3 className="text-sm font-medium">{firstStepTitle}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{traceStatus}</p>
          </div>
          <Link href={uiHref("lens/?tab=traces")} className={traceButtonClass}>
            {tracesReady ? "View traces" : "Set up traces"}
          </Link>
        </li>
        <li
          data-state={activityReady ? "active" : "inactive"}
          className="grid grid-cols-[28px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 py-5 text-muted-foreground data-[state=active]:text-foreground sm:grid-cols-[28px_minmax(0,1fr)_auto]"
        >
          <Step number={2} complete={workerReady} active={waitingForWorker} />
          <div>
            <h3 className="text-sm font-medium">Connect a worker</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              {workerReady ? "Worker connected" : "Runs your investigations in the background."}
            </p>
          </div>
          <Button
            className="col-start-2 w-fit sm:col-start-auto"
            variant={waitingForWorker ? "default" : "ghost"}
            disabled={!canConnect}
            onClick={onConnect}
          >
            {connected ? "Manage worker" : "Connect worker"}
          </Button>
        </li>
        <li
          data-state={workerReady ? "active" : "inactive"}
          className="grid grid-cols-[28px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 py-5 text-muted-foreground data-[state=active]:text-foreground sm:grid-cols-[28px_minmax(0,1fr)_auto]"
        >
          <Step number={3} complete={false} active={workerReady} />
          <div>
            <h3 className="text-sm font-medium">Run an investigation</h3>
          </div>
          <Button
            className="col-start-2 w-fit sm:col-start-auto"
            variant={workerReady ? "default" : "ghost"}
            disabled={!canCreate}
            onClick={onCreate}
          >
            New investigation
          </Button>
        </li>
      </ol>
      {traceError && (
        <div role="alert" className="mt-6 text-sm text-destructive">
          Could not check recorded activity. {traceError}{" "}
          <Button variant="link" onClick={onRetry}>
            Retry
          </Button>
        </div>
      )}
      {readOnly && (
        <p className="mt-6 text-sm text-muted-foreground">An administrator can finish setup and run investigations.</p>
      )}
    </section>
  );
}

function Step({
  number,
  complete,
  checking = false,
  active = false,
}: {
  number: number;
  complete: boolean;
  checking?: boolean;
  active?: boolean;
}) {
  const incomplete = checking ? <Loader2 className="size-4 animate-spin" /> : number;
  const state = stepState(complete, active);
  return (
    <span
      data-state={state}
      className={cn(
        "flex size-7 items-center justify-center rounded-full text-xs font-medium",
        "data-[state=complete]:bg-emerald-50 data-[state=complete]:text-emerald-700 dark:data-[state=complete]:bg-emerald-950 dark:data-[state=complete]:text-emerald-400",
        "data-[state=active]:bg-foreground data-[state=active]:text-background",
        "data-[state=inactive]:border data-[state=inactive]:text-muted-foreground",
      )}
      aria-label={complete ? `Step ${number} complete` : `Step ${number}`}
    >
      {complete ? <Check className="size-4" /> : incomplete}
    </span>
  );
}

function stepState(complete: boolean, active?: boolean) {
  if (complete) return "complete";
  if (active) return "active";
  return "inactive";
}

function activityStatus(traces: boolean, requests: boolean, checking: boolean) {
  if (traces) return "Traces received";
  if (requests) return "Request logs received";
  return checking ? "Checking for activity…" : "Connect your agent and send a trace.";
}
