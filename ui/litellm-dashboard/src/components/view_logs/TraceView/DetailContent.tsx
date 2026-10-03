"use client";
import { useTracesApi } from "@/components/lens/services";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { useState } from "react";
import { AlertTriangle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { type KeyValue, KeyValueRows, objectEntries } from "./KeyValueRows";
import { Card, MessageCard, Section, ToolResultCard } from "./MessageCard";
import type { ErrorSource } from "./traceTree";
import type { Span, SpanDetail, SpanErrorPage, TraceMessage, UIContent, UIMessage } from "./traceTypes";
import { errorSource, parseJson, parseMessages, prettyPayload } from "./traceUtils";

const ERROR_SOURCE_LABEL: Record<ErrorSource, string> = { tool: "Tool", model: "Model", litellm: "LiteLLM" };
const TRACEBACK_MARKER = "Traceback (most recent call last):";
const STATUS_TEXT = "px-5 py-2 text-[13px] tracking-[-0.26px] text-muted-foreground";
const PAYLOAD_PRE =
  "font-mono text-[13px] leading-[1.5] tracking-[-0.26px] break-words whitespace-pre-wrap text-foreground";

/** Exporters record `repr(exc)` + traceback with no separator; keep the exception line. */
export const errorHeadline = (error: string): string =>
  (error.split(TRACEBACK_MARKER, 1)[0].split("\n")[0] ?? "").trim() || error.trim();

/** `ValueError('x not found')` → "ValueError"; plain text → "error". */
const errorReason = (headline: string): string => /^([A-Za-z_][\w.]*)\(/.exec(headline)?.[1] ?? "error";

/** Shared lazy fetch of one span's full input / output / attributes. */
export function useSpanDetail(accessToken: string, traceId: string, spanId: string | null, traceRef?: string) {
  const traces = useTracesApi(accessToken);
  const queryOptions: UseQueryOptions<SpanDetail, Error> = {
    queryKey: ["agentTraceSpan", traceId, traceRef, spanId, accessToken],
    queryFn: () => traces.span(traceId, spanId as string, traceRef),
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
      className="mx-3 mb-3 rounded-[4px] border-[0.67px] border-destructive/40 bg-destructive/5 px-3 py-2.5"
    >
      <div className="flex items-center gap-2 text-[13px] leading-[1.2] font-medium tracking-[-0.26px] text-destructive">
        <AlertTriangle className="size-3.5" />
        {ERROR_SOURCE_LABEL[source]} · {errorReason(headline)}
      </div>
      <pre className={cn("mt-2", PAYLOAD_PRE)}>{headline}</pre>
    </section>
  );
}

function FieldsCard({ entries }: { entries: readonly KeyValue[] }) {
  return (
    <Card className="px-3 py-2.5">
      <KeyValueRows entries={entries} />
    </Card>
  );
}

function TextCard({ text }: { text: string }) {
  return (
    <Card className="px-3 py-2.5">
      <pre className={cn("max-h-96 overflow-auto", PAYLOAD_PRE)}>{text}</pre>
    </Card>
  );
}

function PlainPayload({ value }: { value: string }) {
  const entries = objectEntries(parseJson(value));
  if (entries && entries.length > 0) return <FieldsCard entries={entries} />;
  return <TextCard text={prettyPayload(value)} />;
}

function Messages({ messages, model }: { messages: TraceMessage[]; model: string | null }) {
  return (
    <>
      {messages.map((message, i) => (
        <MessageCard key={`${message.role}-${i}`} message={message} model={model} />
      ))}
    </>
  );
}

const toTraceMessage = (message: UIMessage): TraceMessage => ({
  ...message,
  tool_calls: message.tool_calls?.map((call) => ({
    name: call.name,
    args: parseJson(call.arguments) ?? call.arguments,
  })),
});

interface PayloadProps {
  value: string;
  span: Span;
  role: "input" | "output";
}

const isToolResult = ({ span, role }: Omit<PayloadProps, "value">): boolean =>
  span.type === "tool" && role === "output";

function ToolResult({ value, span }: Omit<PayloadProps, "role">) {
  return <ToolResultCard name={span.name} result={value} failed={span.status === "error"} />;
}

const singleText = (content: UIContent): string | null =>
  content.kind === "messages" && content.messages.length === 1 && !content.messages[0].tool_calls?.length
    ? content.messages[0].content
    : null;

function UIPayload({ content, ...props }: PayloadProps & { content: UIContent }) {
  const toolText = isToolResult(props) ? singleText(content) : null;
  if (toolText !== null) return <ToolResult {...props} value={toolText} />;
  if (content.kind === "messages") {
    return <Messages messages={content.messages.map(toTraceMessage)} model={props.span.model} />;
  }
  if (isToolResult(props)) return <ToolResult {...props} />;
  if (content.kind === "fields" && content.fields.length > 0) {
    return <FieldsCard entries={content.fields.map((field): KeyValue => [field.key, field.value])} />;
  }
  return <TextCard text={content.kind === "text" ? content.text : props.value} />;
}

function Payload(props: PayloadProps) {
  const messages = parseMessages(props.value);
  if (messages) return <Messages messages={messages} model={props.span.model} />;
  if (isToolResult(props)) return <ToolResult {...props} />;
  return <PlainPayload value={props.value} />;
}

function SpanPayload({ content, ...props }: PayloadProps & { content: UIContent | undefined }) {
  return content ? <UIPayload content={content} {...props} /> : <Payload {...props} />;
}

interface DetailContentProps {
  accessToken: string;
  traceId: string;
  traceRef?: string;
  span: Span;
}

function DiagnosticContent({ accessToken, traceId, traceRef, span }: DetailContentProps) {
  const traces = useTracesApi(accessToken);
  const [opened, setOpened] = useState(false);
  const [cursor, setCursor] = useState<string | null>(null);
  const queryOptions: UseQueryOptions<SpanErrorPage, Error> = {
    queryKey: ["agentTraceSpanError", traceId, traceRef, span.span_id, accessToken, cursor],
    queryFn: () => traces.spanError(traceId, span.span_id, { traceRef, cursor }),
    enabled: opened,
    staleTime: Infinity,
    gcTime: 0,
    retry: false,
  };
  const query = useQuery(queryOptions);
  return (
    <section aria-label="Stored diagnostic" className="mx-3 mb-3 space-y-2">
      {span.error_truncated && <p className="text-xs text-muted-foreground">Error preview truncated</p>}
      {!opened && (
        <Button variant="outline" size="sm" onClick={() => setOpened(true)}>
          View stored diagnostic
        </Button>
      )}
      {opened && query.isPending && <p role="status">Loading diagnostic…</p>}
      {opened && query.isError && (
        <div role="alert">
          Could not load diagnostic: {query.error.message}
          <Button variant="outline" size="sm" onClick={() => query.refetch()}>
            Retry
          </Button>
        </div>
      )}
      {opened && query.data && (
        <>
          <TextCard text={query.data.message} />
          <p className="text-xs text-muted-foreground">
            {cursor ? "Continuation" : "Beginning"} of stored diagnostic ({query.data.total_chars.toLocaleString()}{" "}
            characters)
          </p>
          {query.data.next_cursor && (
            <Button variant="outline" size="sm" onClick={() => setCursor(query.data.next_cursor)}>
              Next section
            </Button>
          )}
          {cursor && (
            <Button variant="outline" size="sm" onClick={() => setCursor(null)}>
              Back to beginning
            </Button>
          )}
        </>
      )}
    </section>
  );
}

/** Content tab: the error first (if any), then collapsible Input and Output rendered as chat cards. */
export function DetailContent({ accessToken, traceId, traceRef, span }: DetailContentProps) {
  const detailQuery = useSpanDetail(accessToken, traceId, span.span_id, traceRef);
  const detail = detailQuery.data;
  const empty = detail && !detail.input && !detail.output;
  const inputMessages =
    detail?.input_ui?.kind === "messages"
      ? detail.input_ui.messages.length
      : parseMessages(detail?.input ?? "")?.length;

  return (
    <div className="flex flex-col px-2 pt-1 pb-4">
      <ErrorBlock span={span} />
      {span.error && (
        <DiagnosticContent
          key={`${traceId}:${traceRef}:${span.span_id}`}
          accessToken={accessToken}
          traceId={traceId}
          traceRef={traceRef}
          span={span}
        />
      )}
      {detailQuery.isLoading && <div className={STATUS_TEXT}>Loading span…</div>}
      {detailQuery.isError && <div className={STATUS_TEXT}>Could not load span: {detailQuery.error.message}</div>}
      {detail?.input ? (
        <Section
          key={`${span.span_id}:${inputMessages}`}
          title="Input"
          count={inputMessages}
          defaultOpen={!inputMessages || inputMessages <= 8}
        >
          <SpanPayload value={detail.input} content={detail.input_ui} span={span} role="input" />
        </Section>
      ) : null}
      {detail?.output ? (
        <Section title="Output">
          <SpanPayload value={detail.output} content={detail.output_ui} span={span} role="output" />
        </Section>
      ) : null}
      {empty && span.status !== "error" && (
        <div className="py-12 text-center text-[13px] tracking-[-0.26px] text-muted-foreground">
          No content recorded for this span.
        </div>
      )}
    </div>
  );
}
