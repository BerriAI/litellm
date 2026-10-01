import Link from "next/link";
import { Check, ArrowRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { uiHref } from "@/utils/uiHref";

export function LensWelcome({
  tracesReady,
  checking,
  traceError,
  connected,
  readOnly,
  onConnect,
  onCreate,
  onRetry,
  onExample,
}: {
  tracesReady: boolean;
  checking: boolean;
  traceError?: string;
  connected: boolean;
  readOnly: boolean;
  onConnect: () => void;
  onCreate: () => void;
  onRetry: () => void;
  onExample: () => void;
}) {
  const workerReady = tracesReady && connected;
  const canCreate = workerReady && !readOnly;
  const canConnect = tracesReady && !readOnly;
  const traceStatus = checking ? "Checking for traces…" : "Connect your agent and send a trace.";
  return (
    <section aria-labelledby="lens-welcome" className="max-w-3xl py-4 sm:py-6">
      <h2 id="lens-welcome" className="text-xl font-semibold tracking-tight">
        Find what needs attention
      </h2>
      <p className="mt-2 max-w-lg text-sm leading-6 text-muted-foreground">
        Check how your agents behave. Get findings you can trace back to what happened.
      </p>
      <Button variant="link" className="mt-3 h-auto px-0 text-sm" onClick={onExample}>
        View an example <ArrowRight className="size-3.5" />
      </Button>
      <ol className="mt-8 divide-y border-y">
        <li className="grid grid-cols-[28px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 py-5 sm:grid-cols-[28px_minmax(0,1fr)_auto]">
          <Step number={1} complete={tracesReady} checking={checking} active={!tracesReady} />
          <div>
            <h3 className="text-sm font-medium">Set up traces</h3>
            <p className="mt-1 text-sm text-muted-foreground">{tracesReady ? "Traces received" : traceStatus}</p>
          </div>
          <Link
            href={uiHref("lens/?tab=traces")}
            className={`col-start-2 inline-flex h-9 w-fit items-center justify-center rounded-md px-3 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-ring sm:col-start-auto ${tracesReady ? "hover:bg-muted" : "bg-primary text-primary-foreground hover:bg-primary/90"}`}
          >
            {tracesReady ? "View traces" : "Set up traces"}
          </Link>
        </li>
        <li
          className={`grid grid-cols-[28px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 py-5 sm:grid-cols-[28px_minmax(0,1fr)_auto] ${!tracesReady ? "text-muted-foreground" : ""}`}
        >
          <Step number={2} complete={workerReady} active={tracesReady && !connected} />
          <div>
            <h3 className="text-sm font-medium">Connect a worker</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              {workerReady ? "Worker connected" : "Runs your investigations in the background."}
            </p>
          </div>
          <Button
            className="col-start-2 w-fit sm:col-start-auto"
            variant={tracesReady && !connected ? "default" : "ghost"}
            disabled={!canConnect}
            onClick={onConnect}
          >
            {connected ? "Manage worker" : "Connect worker"}
          </Button>
        </li>
        <li
          className={`grid grid-cols-[28px_minmax(0,1fr)] items-center gap-x-4 gap-y-3 py-5 sm:grid-cols-[28px_minmax(0,1fr)_auto] ${!workerReady ? "text-muted-foreground" : ""}`}
        >
          <Step number={3} complete={false} active={workerReady} />
          <div>
            <h3 className="text-sm font-medium">Run an investigation</h3>
            <p className="mt-1 text-sm text-muted-foreground">Choose activity and what to look for.</p>
          </div>
          <Button className="col-start-2 w-fit sm:col-start-auto" disabled={!canCreate} onClick={onCreate}>
            New investigation
          </Button>
        </li>
      </ol>
      {traceError && (
        <div role="alert" className="mt-6 text-sm text-destructive">
          Could not check traces. {traceError}{" "}
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
  const unfinishedStyle = active ? "bg-foreground text-background" : "border text-muted-foreground";
  return (
    <span
      className={`flex size-7 items-center justify-center rounded-full text-xs font-medium ${complete ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400" : unfinishedStyle}`}
      aria-label={complete ? `Step ${number} complete` : `Step ${number}`}
    >
      {complete ? <Check className="size-4" /> : incomplete}
    </span>
  );
}
