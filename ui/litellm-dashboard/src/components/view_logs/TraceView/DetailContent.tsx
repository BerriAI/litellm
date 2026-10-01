"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";

import { cn } from "@/lib/cva.config";

import { agentTraceSpanCall } from "../../networking";
import { KeyValueRows, objectEntries } from "./KeyValueRows";
import { Card, MessageCard, Section, ToolResultCard } from "./MessageCard";
import type { ErrorSource } from "./traceTree";
import type { Span, SpanDetail } from "./traceTypes";
import { errorSource, parseJson, parseMessages, prettyPayload } from "./traceUtils";

const ERROR_SOURCE_LABEL: Record<ErrorSource, string> = { tool: "Tool", model: "Model", litellm: "LiteLLM" };
const TRACEBACK_MARKER = "Traceback (most recent call last):";
const STATUS_TEXT = "px-5 py-2 text-[13px] tracking-[-0.26px] text-trace-duration";
const PAYLOAD_PRE =
  "font-mono text-[13px] leading-[1.5] tracking-[-0.26px] break-words whitespace-pre-wrap text-trace-text";

/** Exporters record `repr(exc)` + traceback with no separator; keep the exception line. */
export const errorHeadline = (error: string): string =>
  (error.split(TRACEBACK_MARKER, 1)[0].split("\n")[0] ?? "").trim() || error.trim();

/** `ValueError('x not found')` → "ValueError"; plain text → "error". */
const errorReason = (headline: string): string => /^([A-Za-z_][\w.]*)\(/.exec(headline)?.[1] ?? "error";

/** Shared lazy fetch of one span's full input / output / attributes. */
export function useSpanDetail(accessToken: string, traceId: string, spanId: string | null, traceRef?: string) {
  const queryOptions: UseQueryOptions<SpanDetail, Error> = {
    queryKey: ["agentTraceSpan", traceId, traceRef, spanId, accessToken],
    queryFn: () => agentTraceSpanCall(accessToken, traceId, spanId as string, traceRef),
    enabled: spanId !== null,
    staleTime: Infinity,
  };
  return useQuery(queryOptions);
}

export function ErrorBlock({ span }: { span: Span }) {
  const source = errorSource(span);
  if (!source) return null;
  const headline = errorHeadline(span.error ?? "") || "Span reported an error status.";
  return (
    <section
      aria-label="Error"
      className="mx-5 mb-2 rounded-[4px] border-[0.67px] border-destructive/40 bg-destructive/5 px-3 py-2.5"
    >
      <div className="flex items-center gap-2 text-[13px] leading-[1.2] font-medium tracking-[-0.26px] text-destructive">
        <AlertTriangle className="size-3.5" />
        {ERROR_SOURCE_LABEL[source]} · {errorReason(headline)}
      </div>
      <pre className={cn("mt-2", PAYLOAD_PRE)}>{headline}</pre>
    </section>
  );
}

function PlainPayload({ value }: { value: string }) {
  const entries = objectEntries(parseJson(value));
  if (entries && entries.length > 0) {
    return (
      <Card className="px-3 py-2.5">
        <KeyValueRows entries={entries} />
      </Card>
    );
  }
  return (
    <Card className="px-3 py-2.5">
      <pre className={cn("max-h-96 overflow-auto", PAYLOAD_PRE)}>{prettyPayload(value)}</pre>
    </Card>
  );
}

function Payload({ value, span, role }: { value: string; span: Span; role: "input" | "output" }) {
  const messages = parseMessages(value);
  if (messages) {
    return (
      <>
        {messages.map((message, i) => (
          <MessageCard key={`${message.role}-${i}`} message={message} model={span.model} />
        ))}
      </>
    );
  }
  if (span.type === "tool" && role === "output") {
    return <ToolResultCard name={span.name} result={value} failed={span.status === "error"} />;
  }
  return <PlainPayload value={value} />;
}

interface DetailContentProps {
  accessToken: string;
  traceId: string;
  traceRef?: string;
  span: Span;
}

/** Content tab: the error first (if any), then collapsible Input and Output rendered as chat cards. */
export function DetailContent({ accessToken, traceId, traceRef, span }: DetailContentProps) {
  const detailQuery = useSpanDetail(accessToken, traceId, span.span_id, traceRef);
  const detail = detailQuery.data;
  const empty = detail && !detail.input && !detail.output;

  return (
    <div className="flex flex-col px-2 pt-1 pb-4">
      <ErrorBlock span={span} />
      {detailQuery.isLoading && <div className={STATUS_TEXT}>Loading span…</div>}
      {detailQuery.isError && <div className={STATUS_TEXT}>Could not load span: {detailQuery.error.message}</div>}
      {detail?.input ? (
        <Section title="Input">
          <Payload value={detail.input} span={span} role="input" />
        </Section>
      ) : null}
      {detail?.output ? (
        <Section title="Output">
          <Payload value={detail.output} span={span} role="output" />
        </Section>
      ) : null}
      {empty && span.status !== "error" && (
        <div className="py-12 text-center text-[13px] tracking-[-0.26px] text-trace-duration">
          No content recorded for this span.
        </div>
      )}
    </div>
  );
}
