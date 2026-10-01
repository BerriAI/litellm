"use client";

import { CircleCheck } from "lucide-react";

import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";

import { formatCost } from "./AgentTracesTable";
import { errorHeadline } from "./DetailContent";
import { SpanIcon } from "./SpanIcon";
import type { GroupRowData } from "./traceTree";
import type { Span, SpanType } from "./traceTypes";
import { fmtMs, fmtTok } from "./traceUtils";

const HOVER_OPEN_DELAY_MS = 300;
const HOVER_CLOSE_DELAY_MS = 100;

const TYPE_LABEL: Record<SpanType, string> = {
  agent: "Agent",
  llm: "LLM",
  tool: "Tool",
  chain: "Chain",
  framework: "Framework",
};

const ABSOLUTE_TIME_OPTIONS: Intl.DateTimeFormatOptions = {
  month: "2-digit",
  day: "2-digit",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  timeZoneName: "short",
};
const ABSOLUTE_TIME = new Intl.DateTimeFormat("en-US", ABSOLUTE_TIME_OPTIONS);

export const absoluteTime = (traceStartMs: number, offsetMs: number): string =>
  ABSOLUTE_TIME.format(new Date(traceStartMs + offsetMs));

export interface HoverFacts {
  name: string;
  type: SpanType;
  model: string | null;
  startMs: number;
  durationMs: number;
  tokens: number;
  spend: number | null;
  error: string | null;
  failed: boolean;
}

export const spanFacts = (span: Span): HoverFacts => ({
  name: span.name,
  type: span.type,
  model: span.model,
  startMs: span.start_offset_ms,
  durationMs: span.duration_ms,
  tokens: span.input_tokens + span.output_tokens,
  spend: span.spend ?? null,
  error: span.error ?? null,
  failed: span.status === "error",
});

export const groupFacts = (row: GroupRowData): HoverFacts => {
  const start = Math.min(...row.members.map((m) => m.start_offset_ms));
  const end = Math.max(...row.members.map((m) => m.start_offset_ms + m.duration_ms));
  return {
    name: `${row.name} ×${row.members.length}`,
    type: row.type,
    model: row.members[0]?.model ?? null,
    startMs: start,
    durationMs: end - start,
    tokens: row.members.reduce((sum, m) => sum + m.input_tokens + m.output_tokens, 0),
    spend: null,
    error: row.members.find((m) => m.status === "error" && m.error)?.error ?? null,
    failed: row.failedCount > 0,
  };
};

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono text-[11.5px] tabular-nums">{value}</dd>
    </>
  );
}

interface SpanHoverCardProps {
  facts: HoverFacts;
  traceStartMs: number;
  children: React.ReactElement;
}

/** Wraps a tree row so hovering it shows timing, usage and the failure headline. */
export function SpanHoverCard({ facts, traceStartMs, children }: SpanHoverCardProps) {
  return (
    <HoverCard>
      <HoverCardTrigger delay={HOVER_OPEN_DELAY_MS} closeDelay={HOVER_CLOSE_DELAY_MS} render={children} />
      <HoverCardContent side="right" align="start" sideOffset={8} className="w-72 p-3.5 text-[12.5px]">
        <div className="flex items-center gap-2">
          <SpanIcon type={facts.type} model={facts.model} error={facts.failed} />
          <span className="truncate font-semibold">{facts.name}</span>
          {!facts.failed && <CircleCheck className="size-3.5 shrink-0 text-success" aria-label="Success" />}
        </div>
        <div className="mt-0.5 pl-7 text-muted-foreground">
          {TYPE_LABEL[facts.type]}
          {facts.model ? ` · ${facts.model}` : ""}
        </div>
        <div className="mt-3 mb-1 font-semibold">Time</div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          <Fact label="Start" value={absoluteTime(traceStartMs, facts.startMs)} />
          <Fact label="End" value={absoluteTime(traceStartMs, facts.startMs + facts.durationMs)} />
          <Fact label="Duration" value={fmtMs(facts.durationMs)} />
          {facts.tokens > 0 && <Fact label="Tokens" value={fmtTok(facts.tokens)} />}
          {facts.spend != null && <Fact label="Cost" value={formatCost(facts.spend)} />}
        </dl>
        {facts.error && (
          <div className="mt-2.5 rounded-md bg-destructive/10 px-2 py-1.5 font-mono text-[11px] break-words text-destructive">
            {errorHeadline(facts.error)}
          </div>
        )}
      </HoverCardContent>
    </HoverCard>
  );
}
