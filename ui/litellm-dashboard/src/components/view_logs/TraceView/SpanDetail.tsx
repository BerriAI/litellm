"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { cn } from "@/lib/cva.config";

import { agentTraceSpanCall } from "../../networking";
import { SPAN_PILL_CLASS, SpanStatusBadge, SpanTypePill } from "./TracePills";
import type { Span, TraceMessage } from "./traceTypes";
import { fmtMs, parseMessages, prettyPayload, spanLabel } from "./traceUtils";

interface SpanDetailProps {
  accessToken: string;
  traceId: string;
  span: Span;
}

const LONG_MESSAGE_CHARS = 360;

function MessageBlock({ message }: { message: TraceMessage }) {
  const long = (message.content || "").length > LONG_MESSAGE_CHARS;
  const [expanded, setExpanded] = useState(!long);
  return (
    <div className="border-b last:border-b-0">
      <button
        type="button"
        onClick={() => long && setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-[11px] font-semibold uppercase tracking-wide text-muted-foreground"
      >
        <span>
          {message.role}
          {message.name ? ` · ${message.name}` : ""}
        </span>
        {long && (
          <span className="ml-auto font-normal normal-case">{expanded ? "click to collapse" : "click to expand"}</span>
        )}
      </button>
      {message.content && (
        <div
          className={cn(
            "whitespace-pre-wrap break-words px-2.5 pb-2.5 text-[13px]",
            !expanded && "max-h-[88px] overflow-hidden [mask-image:linear-gradient(#000_60%,transparent)]",
          )}
        >
          {message.content}
        </div>
      )}
      {(message.tool_calls ?? []).map((call, i) => (
        <div
          key={`${call.name}-${i}`}
          className={cn(
            "mx-2.5 mb-2.5 whitespace-pre-wrap break-words rounded-md border px-2 py-1.5 font-mono text-xs",
            SPAN_PILL_CLASS.tool,
          )}
        >
          <b>{call.name}</b>({JSON.stringify(call.args, null, 1)})
        </div>
      ))}
    </div>
  );
}

function Payload({ title, value }: { title: string; value: string }) {
  if (!value) return null;
  const messages = parseMessages(value);
  return (
    <section className="mt-3.5">
      <h4 className="mb-1.5 text-xs font-semibold">
        {title}
        {messages && messages.length > 1 ? ` · ${messages.length} messages` : ""}
      </h4>
      {messages ? (
        <div className="overflow-hidden rounded-lg border">
          {messages.map((message, i) => (
            <MessageBlock key={i} message={message} />
          ))}
        </div>
      ) : (
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg border bg-muted px-3 py-2.5 font-mono text-xs">
          {prettyPayload(value)}
        </pre>
      )}
    </section>
  );
}

/** Right pane: span header, joined LiteLLM request, lazily fetched input / output / attributes. */
export function SpanDetail({ accessToken, traceId, span }: SpanDetailProps) {
  const detailQuery = useQuery({
    queryKey: ["agentTraceSpan", traceId, span.span_id, accessToken],
    queryFn: () => agentTraceSpanCall(accessToken, traceId, span.span_id),
    staleTime: Infinity,
  });
  const detail = detailQuery.data;

  return (
    <div className="min-h-0 overflow-auto px-5 pt-4 pb-10" data-testid="span-detail">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <SpanTypePill type={span.type} />
        <span className="text-[15px] font-semibold">{spanLabel(span)}</span>
        <SpanStatusBadge status={span.status} compact />
      </div>
      <div className="mb-3 flex flex-wrap gap-3.5 text-xs text-muted-foreground">
        <span>
          start <b className="font-medium text-foreground">+{fmtMs(span.start_offset_ms)}</b>
        </span>
        <span>
          duration <b className="font-medium text-foreground">{fmtMs(span.duration_ms)}</b>
        </span>
        <span>
          agent <b className="font-medium text-foreground">{span.agent || "—"}</b>
        </span>
        <span>
          span <b className="font-mono font-medium text-foreground">{span.span_id}</b>
        </span>
      </div>
      {span.status === "error" && span.error && (
        <div
          role="alert"
          className="mb-3 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 font-mono text-xs text-destructive"
        >
          {span.error}
        </div>
      )}
      {detailQuery.isLoading && <div className="mt-3.5 text-xs text-muted-foreground">Loading span input/output…</div>}
      {detailQuery.isError && (
        <div className="mt-3.5 text-xs text-destructive">Could not load span details: {detailQuery.error.message}</div>
      )}
      {detail && (
        <>
          <Payload title="Input" value={detail.input} />
          <Payload title="Output" value={detail.output} />
          <section className="mt-3.5">
            <h4 className="mb-1.5 text-xs font-semibold">OTEL attributes</h4>
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg border bg-muted px-3 py-2.5 font-mono text-xs">
              {JSON.stringify(
                { trace_id: traceId, span_id: span.span_id, parent_span_id: span.parent_span_id, ...detail.attributes },
                null,
                2,
              )}
            </pre>
          </section>
        </>
      )}
    </div>
  );
}
