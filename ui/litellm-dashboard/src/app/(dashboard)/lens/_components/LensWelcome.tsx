import { Aperture, ArrowUpRight, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { uiHref } from "@/utils/uiHref";

export function LensWelcome({
  connected,
  readOnly,
  onConnect,
  onCreate,
}: {
  connected: boolean;
  readOnly: boolean;
  onConnect: () => void;
  onCreate: () => void;
}) {
  return (
    <section aria-labelledby="lens-welcome" className="overflow-hidden rounded-xl border bg-card">
      <div className="border-b bg-muted/20 px-6 py-10 sm:px-10">
        <Aperture aria-hidden="true" className="mb-5 size-8" strokeWidth={1.5} />
        <p className="mb-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">Getting started</p>
        <h2 id="lens-welcome" className="text-2xl font-semibold tracking-tight">
          Understand what your agents are doing
        </h2>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
          Tell Lens how your agent should behave. It reviews recorded runs, finds recurring problems, and links each
          finding to the evidence behind it.
        </p>
      </div>
      <ol className="grid divide-y lg:grid-cols-3 lg:divide-x lg:divide-y-0">
        <li className="flex flex-col gap-3 p-6 sm:p-8">
          <span className="flex size-7 items-center justify-center rounded-full border text-xs font-medium">1</span>
          <h3 className="font-medium">Start with recorded activity</h3>
          <p className="text-sm leading-6 text-muted-foreground">
            Use the agent traces or LLM requests already in LiteLLM. Lens needs their inputs and outputs to understand
            what happened.
          </p>
          <a
            href={uiHref("logs/")}
            className="mt-auto inline-flex items-center gap-1 pt-3 text-sm font-medium underline-offset-4 hover:underline"
          >
            View logs <ArrowUpRight aria-hidden="true" className="size-4" />
          </a>
        </li>
        <li className="flex flex-col gap-3 p-6 sm:p-8">
          <span className="flex size-7 items-center justify-center rounded-full border text-xs font-medium">
            {connected ? <CheckCircle2 aria-hidden="true" className="size-4 text-emerald-600" /> : "2"}
          </span>
          <h3 className="font-medium">Connect the analyzer</h3>
          <p className="text-sm leading-6 text-muted-foreground">
            Run one Docker command on your server. The analyzer connects to LiteLLM and runs scans in the background for
            all your lenses.
          </p>
          <div className="mt-auto pt-3">
            {connected && (
              <p role="status" className="text-sm font-medium text-emerald-700 dark:text-emerald-400">
                Analyzer connected
              </p>
            )}
            {!connected && !readOnly && (
              <Button variant="outline" onClick={onConnect}>
                Connect analyzer
              </Button>
            )}
            {!connected && readOnly && (
              <p className="text-sm text-muted-foreground">An administrator can connect the analyzer.</p>
            )}
          </div>
        </li>
        <li className="flex flex-col gap-3 p-6 sm:p-8">
          <span className="flex size-7 items-center justify-center rounded-full border text-xs font-medium">3</span>
          <h3 className="font-medium">Create your first lens</h3>
          <p className="text-sm leading-6 text-muted-foreground">
            Describe expected behavior, choose the runs to review, and start a scan. Run it once or repeat on a
            schedule.
          </p>
          <div className="mt-auto pt-3">
            {!readOnly ? (
              <Button onClick={onCreate}>
                Set up your first lens <ArrowUpRight aria-hidden="true" className="size-4" />
              </Button>
            ) : (
              <p className="text-sm text-muted-foreground">
                Ask an administrator to create a lens. Findings will appear here.
              </p>
            )}
          </div>
        </li>
      </ol>
      <div className="border-t bg-muted/20 px-6 py-4 text-sm text-muted-foreground sm:px-10">
        Try questions like “Did the agent finish the task?”, “Are handoffs working?”, or “Where is it repeating work?”
      </div>
    </section>
  );
}
