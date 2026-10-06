"use client";

import { RefreshCw, Sparkles, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { READABLE_MODEL, type ReadableTurn } from "./readable";

interface ReadableBarProps {
  on: boolean;
  onToggle: () => void;
  rendering: boolean;
  error: Error | null;
  ready: boolean;
  onRegenerate: () => void;
}

export function ReadableBar({ on, onToggle, rendering, error, ready, onRegenerate }: ReadableBarProps) {
  const rendered = on && ready && !rendering;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-end gap-2 text-xs text-muted-foreground">
        {on && rendering && (
          <span role="status" className="inline-flex items-center gap-1.5">
            <RefreshCw className="size-3 animate-spin" />
            Rendering with {READABLE_MODEL}…
          </span>
        )}
        {rendered && (
          <>
            <span className="truncate">Rendered by {READABLE_MODEL}</span>
            <Button variant="ghost" size="xs" onClick={onRegenerate} className="text-muted-foreground">
              <RefreshCw className="size-3" />
              Regenerate
            </Button>
          </>
        )}
        <Button
          variant="outline"
          size="xs"
          aria-pressed={on}
          onClick={onToggle}
          className={cn(on && "border-primary/40 bg-primary/5 text-foreground")}
        >
          <Sparkles className="size-3" />
          Readable
        </Button>
      </div>
      {on && error && (
        <div role="alert" className="flex items-center justify-between gap-3 rounded-md border p-3 text-sm">
          <span className="min-w-0 text-destructive">
            Readable view failed: {error.message}. Showing the recorded thread.
          </span>
          <Button variant="outline" size="sm" onClick={onRegenerate}>
            Retry
          </Button>
        </div>
      )}
    </div>
  );
}

export function ReadableWork({ turn, onOpenStep }: { turn: ReadableTurn; onOpenStep: (id: string) => void }) {
  if (!turn.summary && !turn.steps.length) return null;
  return (
    <div aria-label="Agent work" className="min-w-0 space-y-2 rounded-lg border bg-muted/30 px-3 py-2.5">
      {turn.summary && <p className="text-sm text-muted-foreground">{turn.summary}</p>}
      {turn.steps.length > 0 && (
        <ol className="flex flex-wrap gap-1.5">
          {turn.steps.map((step, index) => (
            <li key={`${step.span_id}-${index}`}>
              <button
                type="button"
                onClick={() => onOpenStep(step.span_id)}
                aria-label={`Inspect ${step.label}`}
                className={cn(
                  "inline-flex max-w-full items-center gap-1 rounded-full border bg-background px-2.5 py-0.5 text-xs hover:bg-muted",
                  step.status === "error" && "border-destructive/40 text-destructive",
                )}
              >
                {step.status === "error" && <TriangleAlert className="size-3 shrink-0" />}
                <span className="truncate">{step.label}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
