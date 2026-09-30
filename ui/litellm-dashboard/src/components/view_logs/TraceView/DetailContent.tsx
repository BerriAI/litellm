"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Bot, CornerDownRight, Wrench } from "lucide-react";

import { agentTraceSpanCall } from "../../networking";
import type { ErrorSource } from "./traceTree";
import type { Span, SpanDetail, TraceMessage } from "./traceTypes";
import { errorSource, parseMessages, prettyPayload } from "./traceUtils";

const ERROR_SOURCE_LABEL: Record<ErrorSource, string> = { tool: "Tool", model: "Model", litellm: "LiteLLM" };
const TRACEBACK_MARKER = "Traceback (most recent call last):";

/** LangSmith records `repr(exc)` + traceback with no separator; keep the exception line. */
export const errorHeadline = (error: string): string =>
  (error.split(TRACEBACK_MARKER, 1)[0].split("\n")[0] ?? "").trim() || error.trim();

/** `ValueError('x not found')` → "ValueError"; plain text → "error". */
const errorReason = (headline: string): string => /^([A-Za-z_][\w.]*)\(/.exec(headline)?.[1] ?? "error";

/** Shared lazy fetch of one span's full input / output / attributes. */
export function useSpanDetail(accessToken: string, traceId: string, spanId: string | null) {
  return useQuery<SpanDetail, Error>({
    queryKey: ["agentTraceSpan", traceId, spanId, accessToken],
    queryFn: () => agentTraceSpanCall(accessToken, traceId, spanId as string),
    enabled: spanId !== null,
    staleTime: Infinity,
  });
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 font-mono text-[9px] font-medium tracking-[0.1em] text-muted-foreground uppercase">
      {children}
    </div>
  );
}

export function TextBlock({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <section>
      <SectionLabel>{label}</SectionLabel>
      <div
        className={`mt-1.5 max-h-72 overflow-auto rounded border border-border bg-card p-3 leading-5 whitespace-pre-wrap break-words text-foreground ${
          mono ? "font-mono text-[11px]" : "text-[12px]"
        }`}
      >
        {value}
      </div>
    </section>
  );
}

function RoleIcon({ role }: { role: string }) {
  if (role === "assistant") return <Bot className="size-3" />;
  if (role === "tool") return <Wrench className="size-3" />;
  return <CornerDownRight className="size-3" />;
}

export function MessageBlock({ message }: { message: TraceMessage }) {
  return (
    <section className="overflow-hidden rounded border border-border bg-card">
      <div className="flex h-7 items-center gap-2 border-b border-border bg-muted/40 px-2.5 font-mono text-[9px] tracking-[0.1em] text-muted-foreground uppercase">
        <RoleIcon role={message.role} />
        {message.role}
        {message.name ? <span className="normal-case">· {message.name}</span> : null}
      </div>
      {(message.tool_calls ?? []).map((call, i) => (
        <div
          key={`${call.name}-${i}`}
          className="border-b border-border bg-muted/20 p-2.5 font-mono text-[11px] break-all text-foreground last:border-b-0"
        >
          {call.name}
          <span className="text-muted-foreground">(</span>
          <span className="text-muted-foreground">{JSON.stringify(call.args)}</span>
          <span className="text-muted-foreground">)</span>
        </div>
      ))}
      {message.content && (
        <div className="p-2.5 text-[12px] leading-5 whitespace-pre-wrap break-words text-foreground">
          {message.content}
        </div>
      )}
    </section>
  );
}

export function ErrorBlock({ span }: { span: Span }) {
  const source = errorSource(span);
  if (!source) return null;
  const headline = errorHeadline(span.error ?? "") || "Span reported an error status.";
  return (
    <section aria-label="Error" className="rounded border border-destructive/40 bg-destructive/5 p-3">
      <div className="flex items-center gap-2 font-mono text-[9px] font-medium tracking-[0.08em] text-destructive uppercase">
        <AlertTriangle className="size-3" />
        {ERROR_SOURCE_LABEL[source]} · {errorReason(headline)}
      </div>
      <pre className="mt-2 font-mono text-[11px] leading-5 whitespace-pre-wrap break-words text-foreground">
        {headline}
      </pre>
    </section>
  );
}

function Payload({ label, value, mono }: { label: string; value: string; mono: boolean }) {
  const messages = parseMessages(value);
  if (messages) {
    return (
      <>
        <SectionLabel>{`${label}${messages.length > 1 ? ` · ${messages.length} messages` : ""}`}</SectionLabel>
        {messages.map((message, i) => (
          <MessageBlock key={`${message.role}-${i}`} message={message} />
        ))}
      </>
    );
  }
  return <TextBlock label={label} value={prettyPayload(value)} mono={mono} />;
}

interface DetailContentProps {
  accessToken: string;
  traceId: string;
  span: Span;
}

/** Content tab: the error first (if any), then what went in and what came out. */
export function DetailContent({ accessToken, traceId, span }: DetailContentProps) {
  const detailQuery = useSpanDetail(accessToken, traceId, span.span_id);
  const detail = detailQuery.data;
  const isTool = span.type === "tool";
  const empty = detail && !detail.input && !detail.output;

  return (
    <div className="space-y-3 p-3">
      <ErrorBlock span={span} />
      {detailQuery.isLoading && <div className="font-mono text-[11px] text-muted-foreground">Loading span…</div>}
      {detailQuery.isError && (
        <div className="font-mono text-[11px] text-muted-foreground">
          Could not load span: {detailQuery.error.message}
        </div>
      )}
      {detail?.input ? <Payload label={isTool ? "Input" : "Input"} value={detail.input} mono={isTool} /> : null}
      {detail?.output ? <Payload label="Output" value={detail.output} mono={isTool} /> : null}
      {empty && span.status !== "error" && (
        <div className="py-12 text-center font-mono text-[11px] text-muted-foreground">
          No content recorded for this span.
        </div>
      )}
    </div>
  );
}
