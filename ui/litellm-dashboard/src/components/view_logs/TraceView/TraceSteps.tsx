"use client";

import { useEffect, useRef } from "react";

import { cn } from "@/lib/cva.config";

import { SPAN_PILL_CLASS, SpanTypePill } from "./TracePills";
import { fmtCost, fmtMs, previewText, spanLabel, type TraceStep } from "./traceUtils";

interface TraceStepsProps {
  steps: TraceStep[];
  selectedId: string | null;
  onSelect: (spanId: string) => void;
}

const SUBAGENT_INDENT_PX = 28;
const BASE_PADDING_PX = 12;

function StepTitle({ step }: { step: TraceStep }) {
  const { span } = step;
  if (span.type === "agent") {
    return (
      <>
        <SpanTypePill type="agent" />
        <span>subagent: {span.name}</span>
        <span className="text-muted-foreground">{fmtMs(span.duration_ms)}</span>
      </>
    );
  }
  const cost = span.type === "llm" ? ` · ${fmtCost(span.litellm?.spend)}` : "";
  return (
    <>
      <SpanTypePill type={span.type} />
      <span className={cn(span.status === "error" && "text-destructive")}>{spanLabel(span)}</span>
      <span className="text-muted-foreground">
        {fmtMs(span.duration_ms)}
        {cost}
      </span>
    </>
  );
}

/** Numbered narrative: every LLM decision and tool result in time order, subagent steps indented. */
export function TraceSteps({ steps, selectedId, onSelect }: TraceStepsProps) {
  const listRef = useRef<HTMLOListElement>(null);

  useEffect(() => {
    if (!selectedId) return;
    const node = listRef.current?.querySelector(`[data-span-id="${selectedId}"]`);
    if (node && "scrollIntoView" in node) (node as HTMLElement).scrollIntoView({ block: "nearest" });
  }, [selectedId]);

  if (steps.length === 0) {
    return <div className="p-4 text-sm text-muted-foreground">This trace has no LLM or tool spans.</div>;
  }

  return (
    <ol ref={listRef} aria-label="Trace steps" className="min-h-0 flex-1 overflow-auto py-1 text-[13px]">
      {steps.map((step) => {
        const pill = step.span.type === "agent" ? "agent" : step.span.type;
        return (
          <li
            key={step.span.span_id}
            data-span-id={step.span.span_id}
            aria-current={step.span.span_id === selectedId ? "step" : undefined}
            onClick={() => onSelect(step.span.span_id)}
            style={{ paddingLeft: BASE_PADDING_PX + step.depth * SUBAGENT_INDENT_PX }}
            className={cn(
              "grid cursor-pointer grid-cols-[22px_1fr] gap-2.5 py-2 pr-3 hover:bg-muted/60",
              step.span.span_id === selectedId && "bg-primary/5",
            )}
          >
            <span
              className={cn(
                "flex size-[22px] items-center justify-center rounded-full border text-[11px] font-semibold",
                SPAN_PILL_CLASS[pill],
              )}
            >
              {step.number}
            </span>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-1.5 font-medium">
                <StepTitle step={step} />
                {step.subagent && <span className="text-[11px] text-muted-foreground">in {step.subagent}</span>}
              </div>
              {step.span.input_preview && (
                <div className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
                  {previewText(step.span.input_preview)}
                </div>
              )}
              {step.toolNames.length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {step.toolNames.map((name, i) => (
                    <span
                      key={`${name}-${i}`}
                      className={cn("rounded border px-1.5 font-mono text-[11px]", SPAN_PILL_CLASS.tool)}
                    >
                      {name}()
                    </span>
                  ))}
                </div>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
