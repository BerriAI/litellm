"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { agentTraceSpanCall } from "../../networking";
import { SPAN_PILL_CLASS, SpanStatusBadge, SpanTypePill } from "./TracePills";
import type { LiteLLMRequest, Span, TraceMessage } from "./traceTypes";
import { fmtCost, fmtMs, parseMessages, prettyPayload, spanLabel } from "./traceUtils";

interface SpanDetailProps {
  accessToken: string;
  traceId: string;
  span: Span;
  /** Open the existing request-log drawer for a LiteLLM request id. */
  onOpenRequestLog?: (requestId: string) => void;
}

const LONG_MESSAGE_CHARS = 360;

function Field({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className={cn("truncate font-medium", mono && "font-mono text-xs")} title={value}>
        {value || "—"}
      </div>
    </div>
  );
}

function TokenFlowBar({ request }: { request: LiteLLMRequest }) {
  const fresh = Math.max(request.prompt_tokens - request.cache_read_tokens - request.cache_write_tokens, 0);
  const total = request.prompt_tokens + request.completion_tokens || 1;
  const parts = [
    { label: "input", value: fresh, className: "bg-info" },
    { label: "cache read", value: request.cache_read_tokens, className: "bg-cyan-500" },
    { label: "cache write", value: request.cache_write_tokens, className: "bg-violet-500" },
    { label: "output", value: request.completion_tokens, className: "bg-amber-500" },
  ];
  return (
    <>
      <div className="mt-2.5 flex h-2 overflow-hidden rounded bg-muted">
        {parts.map((part) => (
          <i key={part.label} className={part.className} style={{ width: `${(part.value / total) * 100}%` }} />
        ))}
      </div>
      <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted-foreground">
        {parts.map((part) => (
          <span key={part.label} className="inline-flex items-center gap-1">
            <i className={cn("inline-block size-2 rounded-sm", part.className)} />
            {part.label} {part.value.toLocaleString()}
          </span>
        ))}
      </div>
    </>
  );
}

function LiteLLMRequestCard({
  request,
  onOpenRequestLog,
}: {
  request: LiteLLMRequest;
  onOpenRequestLog?: (requestId: string) => void;
}) {
  return (
    <section
      aria-label="LiteLLM request"
      className="rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2.5 dark:border-cyan-900 dark:bg-cyan-950/40"
    >
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold text-cyan-700 dark:text-cyan-300">
          LiteLLM request · joined from spend log
        </span>
        {onOpenRequestLog && (
          <Button variant="outline" size="xs" onClick={() => onOpenRequestLog(request.request_id)}>
            Open request log →
          </Button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-3.5 gap-y-2 sm:grid-cols-3">
        <Field label="Request ID" value={request.request_id} mono />
        <Field label="Model" value={request.model_group || request.model} />
        <Field label="Deployment" value={request.provider ? `${request.model} (${request.provider})` : request.model} />
        <Field
          label="Tokens"
          value={`${request.prompt_tokens.toLocaleString()} → ${request.completion_tokens.toLocaleString()}`}
        />
        <Field label="Cost" value={fmtCost(request.spend)} />
        <Field label="Status" value={request.status} />
        <Field label="Virtual key" value={request.key_alias} />
        <Field label="Team" value={request.team_alias} />
        <Field
          label="Latency"
          value={`${fmtMs(request.latency_ms)}${request.ttft_ms != null ? ` · TTFT ${fmtMs(request.ttft_ms)}` : ""}`}
        />
      </div>
      <TokenFlowBar request={request} />
    </section>
  );
}

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
export function SpanDetail({ accessToken, traceId, span, onOpenRequestLog }: SpanDetailProps) {
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
      {span.litellm && <LiteLLMRequestCard request={span.litellm} onOpenRequestLog={onOpenRequestLog} />}
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
