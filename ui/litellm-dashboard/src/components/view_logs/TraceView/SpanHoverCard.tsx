"use client";

import { Check } from "lucide-react";

import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";

import { formatCost } from "./AgentTracesTable";
import { errorHeadline } from "./DetailContent";
import { SpanIcon } from "./SpanIcon";
import type { GroupRowData } from "./traceTree";
import type { Span, SpanType } from "./traceTypes";
import { fmtMs, fmtTok } from "./traceUtils";

export const HOVER_OPEN_DELAY_MS = 300;
const HOVER_CLOSE_DELAY_MS = 100;

const TYPE_LABEL: Record<SpanType, string> = {
  agent: "Agent",
  llm: "LLM",
  tool: "Tool",
  chain: "Chain",
  framework: "Framework",
  retriever: "Retriever",
  embedding: "Embedding",
  reranker: "Reranker",
  guardrail: "Guardrail",
  evaluator: "Evaluator",
  prompt: "Prompt",
  decision: "Decision",
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
  tags: readonly string[];
}

const agentTags = (agent: string): readonly string[] => (agent ? [`agent:${agent}`] : []);

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
  tags: agentTags(span.agent),
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
    tags: agentTags(row.agent),
  };
};

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-trace-duration">{label}</dt>
      <dd className="font-mono text-trace-text tabular-nums">{value}</dd>
    </div>
  );
}

function FactSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section aria-label={title} className="flex flex-col gap-1.5">
      <h4 className="font-semibold text-trace-text">{title}</h4>
      {children}
    </section>
  );
}

function HoverCardBody({ facts, traceStartMs }: { facts: HoverFacts; traceStartMs: number }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start gap-2">
        <span className="mt-0.5">
          <SpanIcon type={facts.type} model={facts.model} error={facts.failed} size="sm" />
        </span>
        <div className="flex min-w-0 flex-col">
          <div className="flex min-w-0 items-center gap-2">
            <span className="truncate font-semibold text-trace-text">{facts.name}</span>
            {!facts.failed && (
              <span
                className="grid size-4 shrink-0 place-items-center rounded-full bg-trace-ok p-0.5"
                aria-label="Success"
              >
                <Check className="size-3 text-trace-ok-glyph" strokeWidth={2.5} />
              </span>
            )}
          </div>
          <span className="text-[12px] text-trace-duration">
            {TYPE_LABEL[facts.type]}
            {facts.model ? ` · ${facts.model}` : ""}
          </span>
        </div>
      </div>
      <FactSection title="Time">
        <dl className="flex flex-col gap-1.5">
          <Fact label="Start" value={absoluteTime(traceStartMs, facts.startMs)} />
          <Fact label="End" value={absoluteTime(traceStartMs, facts.startMs + facts.durationMs)} />
          <Fact label="Duration" value={fmtMs(facts.durationMs)} />
        </dl>
      </FactSection>
      {(facts.tokens > 0 || facts.spend != null) && (
        <FactSection title="Usage">
          <dl className="flex flex-col gap-1.5">
            {facts.tokens > 0 && <Fact label="Tokens" value={fmtTok(facts.tokens)} />}
            {facts.spend != null && <Fact label="Cost" value={formatCost(facts.spend)} />}
          </dl>
        </FactSection>
      )}
      {facts.tags.length > 0 && (
        <FactSection title="Tags">
          <ul className="flex flex-wrap gap-1">
            {facts.tags.map((tag) => (
              <li key={tag} className="rounded-[3px] bg-trace-tag px-1 py-0.5 text-trace-duration">
                {tag}
              </li>
            ))}
          </ul>
        </FactSection>
      )}
      {facts.error && (
        <div className="rounded-[4px] bg-destructive/10 px-2 py-1.5 font-mono text-[12px] break-words text-destructive">
          {errorHeadline(facts.error)}
        </div>
      )}
    </div>
  );
}

interface SpanHoverCardProps {
  facts: HoverFacts;
  traceStartMs: number;
  children: React.ReactElement;
}

/** Wraps a tree row: after a hover delay, shows timing, usage, tags and the failure headline beside the drawer. */
export function SpanHoverCard({ facts, traceStartMs, children }: SpanHoverCardProps) {
  return (
    <HoverCard>
      <HoverCardTrigger delay={HOVER_OPEN_DELAY_MS} closeDelay={HOVER_CLOSE_DELAY_MS} render={children} />
      <HoverCardContent
        side="left"
        align="start"
        sideOffset={0}
        alignOffset={0}
        data-testid="span-hover-card"
        className="max-h-[min(520px,calc(100vh-16px))] w-auto max-w-[420px] overflow-y-auto rounded-[6px] border border-trace-border bg-trace-surface p-3 text-[13px] leading-[1.2] tracking-[-0.26px] text-trace-text shadow-trace-md ring-0 transition-opacity duration-150 ease-[cubic-bezier(0,0,0.2,1)] data-closed:animate-none data-ending-style:opacity-0 data-open:animate-none motion-reduce:transition-none"
      >
        <HoverCardBody facts={facts} traceStartMs={traceStartMs} />
      </HoverCardContent>
    </HoverCard>
  );
}
