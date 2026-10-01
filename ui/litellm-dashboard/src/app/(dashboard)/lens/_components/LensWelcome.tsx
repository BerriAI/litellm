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
    <section aria-labelledby="lens-welcome" className="flex flex-col items-center px-6 py-24 text-center">
      <h2 id="lens-welcome" className="text-xl font-semibold tracking-tight">
        Understand what your agents are doing
      </h2>
      <p className="mt-3 max-w-lg text-sm leading-6 text-muted-foreground">
        Choose the runs to review and describe what you expect. Lens finds recurring issues and links them to the
        evidence.
      </p>
      {!readOnly ? (
        <Button className="mt-6" onClick={onCreate}>
          Set up your first lens
        </Button>
      ) : (
        <p className="mt-6 text-sm text-muted-foreground">
          Ask an administrator to create a lens. Findings will appear here.
        </p>
      )}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-sm">
        <a href={uiHref("logs/")} className="text-muted-foreground underline-offset-4 hover:underline">
          View logs
        </a>
        {connected && (
          <p role="status" className="text-muted-foreground">
            Analyzer connected
          </p>
        )}
        {!connected && !readOnly && (
          <Button variant="link" className="h-auto p-0 font-normal text-muted-foreground" onClick={onConnect}>
            Connect analyzer
          </Button>
        )}
        {!connected && readOnly && <p className="text-muted-foreground">An administrator can connect the analyzer.</p>}
      </div>
    </section>
  );
}
