import { useState } from "react";
import { Button } from "@/components/ui/button";
import type { Trace } from "../../types";

export function CaptureCoverage({ capture }: { capture: Trace["capture"] }) {
  const [limit, setLimit] = useState(20);
  if (!capture) return null;
  const missing = capture.actors.filter((actor) => actor.llm_calls > 0 && !actor.reply_events && !actor.model_outputs);
  const names = missing
    .slice(0, 3)
    .map((actor) => actor.name)
    .join(", ");
  return (
    <section aria-label="Capture coverage" className="rounded-md border p-3 text-sm">
      <p className="font-medium">Capture coverage</p>
      <p className="text-muted-foreground">
        {capture.content_events} content events received across {capture.actors.length} actors. More records may still
        arrive.
      </p>
      {missing.length > 0 && (
        <p role="status">
          No reply recorded yet for {names}
          {missing.length > 3 ? ` and ${missing.length - 3} more actors` : ""}.
        </p>
      )}
      {capture.unassigned_events > 0 && (
        <p role="status">{capture.unassigned_events} content events have unconfirmed actor ownership.</p>
      )}
      {capture.warning_events > 0 && (
        <p role="status">{capture.warning_events} records carry capture warnings. Inspect their source spans.</p>
      )}
      <details className="mt-2">
        <summary className="cursor-pointer">View actor coverage</summary>
        <ul className="mt-2 space-y-2">
          {capture.actors.slice(0, limit).map((actor) => (
            <li key={actor.actor_id}>
              <span className="font-medium">{actor.name}</span>: {actor.llm_calls} model calls, {actor.tool_calls}{" "}
              tools, {actor.reply_events} reply events, {actor.model_outputs} recorded model outputs
            </li>
          ))}
        </ul>
        {capture.actors.length > limit && (
          <Button variant="outline" size="sm" className="mt-2" onClick={() => setLimit((value) => value + 20)}>
            Show more actors
          </Button>
        )}
      </details>
    </section>
  );
}
