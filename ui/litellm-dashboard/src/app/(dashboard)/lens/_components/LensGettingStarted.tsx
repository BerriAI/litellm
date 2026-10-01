import { Aperture, ArrowUpRight, CheckCircle2, Circle } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { uiHref } from "@/utils/uiHref";

export function LensGettingStarted({
  tracingEnabled,
  connected,
  readOnly,
  onConnect,
  onCreate,
}: {
  tracingEnabled: boolean;
  connected: boolean;
  readOnly: boolean;
  onConnect: () => void;
  onCreate: () => void;
}) {
  return (
    <section aria-labelledby="lens-getting-started" className="mx-auto max-w-5xl rounded-xl border bg-card">
      <div className="border-b px-6 py-8 sm:px-8">
        <div className="mb-5 flex items-center gap-3">
          <div className="rounded-lg border bg-muted/40 p-2.5">
            <Aperture aria-hidden="true" className="size-6" strokeWidth={1.75} />
          </div>
          <span className="text-sm text-muted-foreground">No lenses available yet</span>
        </div>
        <h2 id="lens-getting-started" className="text-2xl font-semibold tracking-tight">
          Turn agent activity into answers
        </h2>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
          A lens reviews recorded activity for the questions you care about. Find recurring failures, unnecessary work,
          and useful patterns, with evidence linked to the original runs.
        </p>
      </div>
      <ol className="grid gap-6 px-6 py-7 sm:px-8 lg:grid-cols-3 lg:gap-8">
        <li className="flex flex-col items-start gap-3">
          <span className="flex size-7 items-center justify-center rounded-full border text-xs font-medium">1</span>
          <h3 className="text-sm font-semibold">Record activity</h3>
          <p className="text-sm leading-6 text-muted-foreground">
            Enable tracing, then record agent runs or LLM requests. Open the Agent Traces tab in Logs for setup and
            check that your activity is arriving.
          </p>
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            {tracingEnabled ? (
              <CheckCircle2 aria-hidden="true" className="size-3.5 text-emerald-600" />
            ) : (
              <Circle aria-hidden="true" className="size-3.5" />
            )}
            {tracingEnabled ? "Tracing configured" : "Tracing not configured"}
          </p>
          <a href={uiHref("logs")} className={buttonVariants({ variant: "outline", className: "mt-auto" })}>
            Open logs
            <ArrowUpRight aria-hidden="true" className="size-4" />
          </a>
        </li>
        <li className="flex flex-col items-start gap-3">
          <span className="flex size-7 items-center justify-center rounded-full border text-xs font-medium">2</span>
          <h3 className="text-sm font-semibold">Connect an analyzer</h3>
          <p className="text-sm leading-6 text-muted-foreground">
            Run the Lens analyzer with Docker on your server. It connects to this deployment to review activity and save
            findings in the background.
          </p>
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            {connected ? (
              <CheckCircle2 aria-hidden="true" className="size-3.5 text-emerald-600" />
            ) : (
              <Circle aria-hidden="true" className="size-3.5" />
            )}
            {connected ? "Analyzer connected" : "Analyzer not connected"}
          </p>
          {!readOnly && (
            <Button variant="outline" className="mt-auto" onClick={onConnect}>
              {connected ? "Manage analyzer" : "Connect analyzer"}
            </Button>
          )}
        </li>
        <li className="flex flex-col items-start gap-3">
          <span className="flex size-7 items-center justify-center rounded-full border text-xs font-medium">3</span>
          <h3 className="text-sm font-semibold">Create your first lens</h3>
          <p className="text-sm leading-6 text-muted-foreground">
            Choose the activity and questions to review, then select an analysis model and budget. Run once or keep
            monitoring for new findings.
          </p>
          {!readOnly && (
            <Button className="mt-auto" onClick={onCreate}>
              Create a lens
              <ArrowUpRight aria-hidden="true" className="size-4" />
            </Button>
          )}
        </li>
      </ol>
      {readOnly && (
        <p className="mx-6 mb-6 rounded-lg border bg-muted/40 p-4 text-sm leading-6 sm:mx-8">
          You have read-only access. Ask a proxy admin to connect an analyzer and create a lens for the activity you can
          access.
        </p>
      )}
      <div className="rounded-b-xl border-t bg-muted/20 px-6 py-6 sm:px-8">
        <h3 className="text-sm font-medium">Start with a question</h3>
        <ul className="mt-3 grid gap-3 text-sm leading-6 text-muted-foreground lg:grid-cols-3 lg:gap-8">
          <li>Where do agents get stuck or repeat the same work?</li>
          <li>Which answers make claims without supporting evidence?</li>
          <li>What are people asking our agents to do?</li>
        </ul>
      </div>
    </section>
  );
}
