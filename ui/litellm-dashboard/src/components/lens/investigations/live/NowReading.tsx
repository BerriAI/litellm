"use client";

import { cn } from "@/lib/cva.config";

import {
  activityOperation,
  activityPhase,
  durationLabel,
  outcome,
  shortVerdict,
  toolCallSummary,
} from "../../model/live";
import { laneText, typedChars, type Lane } from "../../model/stage";
import type { Activity } from "../../model/types";
import { useNow } from "@/hooks/useNow";
import { ModelName } from "./LiveStrip";

const RED = "text-destructive";

function dotTone(reading: boolean, issueShown: boolean): string {
  if (reading) return "bg-foreground motion-safe:animate-pulse";
  return issueShown ? "bg-destructive" : "bg-foreground/50";
}

function LaneRow({ lane, now, charMs }: { lane: Lane; now: number; charMs: number }) {
  const { review } = lane;
  const text = review ? laneText(review.reasoning) : "";
  const typed = typedChars(lane, now, charMs);
  const typing = !!review && typed < text.length;
  const issue = review ? outcome(review) === "issue" : false;
  return (
    <li className="flex flex-col gap-1 rounded-lg bg-background/70 px-2.5 py-2 text-xs ring-1 ring-border/60 motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-top-1 motion-safe:duration-200">
      <div className="flex items-center gap-2">
        <span aria-hidden="true" className={cn("size-1.5 shrink-0 rounded-full", dotTone(!review, issue && !typing))} />
        <span className="truncate font-medium text-foreground">{lane.agent || "trace"}</span>
        <span className="truncate font-mono text-xs text-muted-foreground">{lane.traceId.slice(0, 8)}</span>
        <span className="ml-auto shrink-0 text-xs tabular-nums text-muted-foreground">
          {review ? durationLabel(review.duration_ms) : `reading · ${durationLabel(Math.max(0, now - lane.startedAt))}`}
        </span>
      </div>
      {!review ? (
        <div aria-hidden="true" className="h-0.5 overflow-hidden rounded-full bg-muted">
          <div className="h-full w-2/5 rounded-full bg-foreground/30 motion-safe:animate-lens-shimmer" />
        </div>
      ) : (
        <>
          <p className="min-h-[1.25rem] leading-relaxed text-muted-foreground">
            {text.slice(0, typed)}
            {typing && (
              <span aria-hidden="true" className="ml-px inline-block h-3 w-px translate-y-0.5 bg-foreground" />
            )}
          </p>
          {!typing && (
            <p
              className={cn(
                "truncate font-medium motion-safe:animate-in motion-safe:fade-in",
                issue ? RED : "text-foreground",
              )}
            >
              {shortVerdict(review)}
            </p>
          )}
        </>
      )}
    </li>
  );
}

export function NowReading({
  model,
  counter,
  lanes,
  now,
  charMs,
}: {
  model: string;
  counter: string;
  lanes: readonly Lane[];
  now: number;
  charMs: number;
}) {
  return (
    <section aria-label="Now reading" className="mb-3 flex flex-col gap-2 rounded-xl bg-muted p-2.5 ring-1 ring-border">
      <header className="flex items-center justify-between gap-3 px-0.5">
        <ModelName model={model} size="md" />
        <span role="status" className="text-xs tabular-nums text-muted-foreground">
          {counter}
        </span>
      </header>
      {lanes.length ? (
        <ol className="flex flex-col gap-1.5">
          {lanes.map((lane) => (
            <LaneRow key={lane.key} lane={lane} now={now} charMs={charMs} />
          ))}
        </ol>
      ) : (
        <div aria-hidden="true" className="h-0.5 overflow-hidden rounded-full bg-background/60">
          <div className="h-full w-2/5 rounded-full bg-foreground/30 motion-safe:animate-lens-shimmer" />
        </div>
      )}
    </section>
  );
}

export function ActiveWork({ model, activities }: { model: string; activities: readonly Activity[] }) {
  const now = useNow(500);
  return (
    <section
      aria-label="Current work"
      className="mb-3 flex flex-col gap-2 rounded-xl bg-muted p-2.5 ring-1 ring-border"
    >
      <header className="flex items-center justify-between gap-3 px-0.5">
        <ModelName model={model} size="md" />
        <span className="text-xs tabular-nums text-muted-foreground">{activities.length} active</span>
      </header>
      <ol aria-label="Active analysis tasks" className="flex flex-col gap-1.5">
        {activities.map((activity) => {
          const tools = toolCallSummary(activity.tool_calls);
          return (
            <li
              key={activity.id}
              className="flex flex-col gap-1 rounded-lg bg-background/70 px-2.5 py-2 text-xs ring-1 ring-border/60"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="font-medium text-foreground">{activity.label || activityPhase(activity)}</span>
                <span className="shrink-0 tabular-nums text-muted-foreground">
                  {durationLabel(Math.max(0, now - Date.parse(activity.started_at)))}
                </span>
              </div>
              <p className="text-muted-foreground">
                {activityPhase(activity)}
                {activity.execution_ids.length > 0 &&
                  ` · ${activity.execution_ids.length} ${activity.execution_ids.length === 1 ? "trace" : "traces"}`}
              </p>
              <p role="status" className="text-foreground">
                {activityOperation(activity)}
              </p>
              {tools && <p className="text-muted-foreground">Tool calls: {tools}</p>}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
