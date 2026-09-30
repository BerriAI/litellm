"use client";

import { useQuery } from "@tanstack/react-query";
import { Copy } from "lucide-react";
import { useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";

import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { agentTraceMarkdownUrl, agentTraceSpanCall } from "../../networking";
import type { LiteLLMRequest, Span, TraceMessage } from "./traceTypes";
import {
  cleanError,
  compactPayload,
  copyStepText,
  fmtCost,
  fmtMs,
  parseMessages,
  type OutlineRow,
} from "./traceUtils";

interface SpanDetailProps {
  accessToken: string;
  traceId: string;
  row: OutlineRow;
  /** Open the existing request-log drawer for a LiteLLM request id. */
  onOpenRequestLog?: (requestId: string) => void;
}

const LONG_MESSAGE_CHARS = 600;
/** Earlier LLM input messages are folded behind "Show N earlier". */
const RECENT_MESSAGES = 3;
const TOOL_ARGS_MAX_CHARS = 280;

/* ------------------------------------------------------------------ */
/*  Small typographic primitives                                       */
/* ------------------------------------------------------------------ */

function SectionLabel({ children }: { children: ReactNode }) {
  return <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{children}</div>;
}

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <section className="mb-6" aria-label={label}>
      <SectionLabel>{label}</SectionLabel>
      {children}
    </section>
  );
}

function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <pre
      className={cn(
        "max-h-96 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-foreground",
        className,
      )}
    >
      {children}
    </pre>
  );
}

/** Exception text: default foreground, set apart by a thin neutral rule (no red box). */
function ErrorText({ error }: { error: string }) {
  return (
    <section className="mb-6" aria-label="Error" role="alert">
      <SectionLabel>Error</SectionLabel>
      <Mono className="border-l-2 border-foreground/20 pl-3">{error}</Mono>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/*  Conversation                                                       */
/* ------------------------------------------------------------------ */

/** Markdown with a quiet, document-like rhythm (no colors, no boxes). */
const MARKDOWN_CLASS = [
  "text-sm leading-relaxed break-words",
  "[&_h1]:mt-4 [&_h1]:mb-2 [&_h1]:text-[15px] [&_h1]:font-semibold",
  "[&_h2]:mt-4 [&_h2]:mb-2 [&_h2]:text-[15px] [&_h2]:font-semibold",
  "[&_h3]:mt-3 [&_h3]:mb-1.5 [&_h3]:text-sm [&_h3]:font-semibold",
  "[&_p]:my-2 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5",
  "[&_code]:font-mono [&_code]:text-xs [&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-muted [&_pre]:p-3",
  "[&_a]:underline [&_a]:underline-offset-4 [&_strong]:font-semibold [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
].join(" ");

function MessageText({ text }: { text: string }) {
  const long = text.length > LONG_MESSAGE_CHARS;
  const [expanded, setExpanded] = useState(false);
  return (
    <>
      <div className={cn(MARKDOWN_CLASS, long && !expanded && "max-h-60 overflow-hidden [mask-image:linear-gradient(#000_70%,transparent)]")}>
        <ReactMarkdown>{text}</ReactMarkdown>
      </div>
      {long && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="mt-1 text-xs text-muted-foreground hover:text-foreground"
        >
          {expanded ? "Show less" : "Show all"}
        </button>
      )}
    </>
  );
}

const clipArgs = (args: unknown): string => {
  const text = JSON.stringify(args) ?? "";
  return text.length > TOOL_ARGS_MAX_CHARS ? `${text.slice(0, TOOL_ARGS_MAX_CHARS)}…` : text;
};

function Message({ message }: { message: TraceMessage }) {
  return (
    <div className="py-3 first:pt-0">
      <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {message.role}
        {message.name ? ` · ${message.name}` : ""}
      </div>
      {message.content && <MessageText text={message.content} />}
      {(message.tool_calls ?? []).map((call, i) => (
        <div key={`${call.name}-${i}`} className="mt-1 break-words font-mono text-xs leading-relaxed">
          <span className="text-muted-foreground">→ </span>
          {call.name}
          <span className="text-muted-foreground">({clipArgs(call.args)})</span>
        </div>
      ))}
      {!message.content && !message.tool_calls?.length && <div className="text-sm text-muted-foreground">(empty)</div>}
    </div>
  );
}

function Conversation({ messages, foldEarlier }: { messages: TraceMessage[]; foldEarlier: boolean }) {
  const [showAll, setShowAll] = useState(false);
  const hidden = foldEarlier && !showAll ? Math.max(messages.length - RECENT_MESSAGES, 0) : 0;
  return (
    <div className="divide-y divide-border/60">
      {hidden > 0 && (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="block pb-3 text-xs text-muted-foreground hover:text-foreground"
        >
          Show {hidden} earlier message{hidden === 1 ? "" : "s"}
        </button>
      )}
      {messages.slice(hidden).map((message, i) => (
        <Message key={hidden + i} message={message} />
      ))}
    </div>
  );
}

/** Messages when the payload is a message list, else compact JSON / text. */
function Payload({ value, foldEarlier = false }: { value: string; foldEarlier?: boolean }) {
  if (!value || value === "{}" || value === "null") return <div className="text-sm text-muted-foreground">(empty)</div>;
  const messages = parseMessages(value);
  if (messages) return <Conversation messages={messages} foldEarlier={foldEarlier} />;
  return <Mono>{compactPayload(value)}</Mono>;
}

/* ------------------------------------------------------------------ */
/*  Request tab                                                        */
/* ------------------------------------------------------------------ */

function DefinitionList({ items }: { items: { label: string; value: ReactNode; mono?: boolean }[] }) {
  return (
    <dl className="grid grid-cols-[minmax(96px,max-content)_1fr] gap-x-6 text-[13px]">
      {items.map((item) => (
        <div key={item.label} className="contents">
          <dt className="border-b border-border/60 py-2 text-muted-foreground">{item.label}</dt>
          <dd
            className={cn(
              "min-w-0 border-b border-border/60 py-2 break-words tabular-nums",
              item.mono && "font-mono text-xs",
            )}
          >
            {item.value || "—"}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function RequestFacts({
  request,
  onOpenRequestLog,
}: {
  request: LiteLLMRequest;
  onOpenRequestLog?: (requestId: string) => void;
}) {
  const cache = `${request.cache_read_tokens.toLocaleString()} / ${request.cache_write_tokens.toLocaleString()}`;
  return (
    <section aria-label="LiteLLM request">
      <DefinitionList
        items={[
          { label: "Model", value: request.model_group || request.model },
          { label: "Deployment", value: request.model, mono: true },
          { label: "Provider", value: request.provider },
          { label: "Key", value: request.key_alias },
          { label: "Team", value: request.team_alias },
          {
            label: "Tokens",
            value: `${request.prompt_tokens.toLocaleString()} → ${request.completion_tokens.toLocaleString()}`,
          },
          { label: "Cache read / write", value: cache },
          { label: "Cost", value: fmtCost(request.spend) },
          { label: "Latency", value: fmtMs(request.latency_ms) },
          { label: "TTFT", value: request.ttft_ms != null ? fmtMs(request.ttft_ms) : "" },
          {
            label: "Request ID",
            mono: true,
            value: (
              <span className="inline-flex items-center gap-1">
                {request.request_id}
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label="Copy request id"
                  onClick={() => void copyToClipboard(request.request_id, "Request ID copied")}
                >
                  <Copy />
                </Button>
              </span>
            ),
          },
        ]}
      />
      {onOpenRequestLog && (
        <button
          type="button"
          onClick={() => onOpenRequestLog(request.request_id)}
          className="mt-4 text-[13px] text-foreground underline decoration-border underline-offset-4 hover:decoration-foreground"
        >
          Open request log ↗
        </button>
      )}
    </section>
  );
}

function Attributes({ attributes }: { attributes: Record<string, string> }) {
  const entries = Object.entries(attributes).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) return <div className="text-sm text-muted-foreground">No attributes.</div>;
  return (
    <dl className="text-xs" aria-label="OTEL attributes">
      {entries.map(([key, value]) => (
        <div key={key} className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-4 border-b border-border/60 py-1.5">
          <dt className="truncate text-muted-foreground" title={key}>
            {key}
          </dt>
          <dd className="break-words font-mono">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/* ------------------------------------------------------------------ */
/*  Content per row kind                                               */
/* ------------------------------------------------------------------ */

function useSpanDetail(accessToken: string, traceId: string, span: Span | undefined) {
  return useQuery({
    queryKey: ["agentTraceSpan", traceId, span?.span_id, accessToken],
    queryFn: () => agentTraceSpanCall(accessToken, traceId, span?.span_id as string),
    enabled: span !== undefined,
    staleTime: Infinity,
  });
}

function SpanContent({ row, detail }: { row: OutlineRow; detail: { input: string; output: string } }) {
  const span = row.span as Span;
  if (row.kind === "input") return <Payload value={detail.input} />;
  if (row.kind === "output") return <Payload value={detail.output} />;
  if (span.type === "tool") {
    return (
      <>
        <Section label="Args">
          <Mono>{compactPayload(detail.input || span.input_preview)}</Mono>
        </Section>
        <Section label="Result">
          <Payload value={detail.output} />
        </Section>
      </>
    );
  }
  if (span.type === "llm") {
    const messages = [...(parseMessages(detail.input) ?? []), ...(parseMessages(detail.output) ?? [])];
    if (messages.length > 0) return <Conversation messages={messages} foldEarlier />;
  }
  return (
    <>
      <Section label="Task">
        <Payload value={detail.input} />
      </Section>
      <Section label="Result">
        <Payload value={detail.output} />
      </Section>
    </>
  );
}

function GroupContent({ row }: { row: OutlineRow }) {
  if (row.fold) {
    const { fold } = row;
    return (
      <DefinitionList
        items={[
          { label: "Invocations", value: fold.invocations.length.toLocaleString() },
          { label: "Failed", value: fold.failed ? `${fold.failed} with a failed step` : "none" },
          { label: "Cost", value: fmtCost(fold.spend) },
          { label: "Median duration", value: fmtMs(fold.p50Ms) },
        ]}
      />
    );
  }
  const failed = (row.failures?.nodes ?? []).flatMap((n) =>
    n.kind === "span" && n.span.status === "error" ? [n.span] : [],
  );
  return (
    <Section label={`${failed.length} failed calls`}>
      <div className="divide-y divide-border/60">
        {failed.map((span, i) => (
          <div key={span.span_id} className="py-2.5 first:pt-0">
            <div className="mb-1 text-xs text-muted-foreground tabular-nums">
              #{i + 1} · +{fmtMs(span.start_offset_ms)}
            </div>
            <Mono>{compactPayload(span.input_preview) || "(no args)"}</Mono>
            {cleanError(span.error) && (
              <Mono className="mt-1 border-l-2 border-foreground/20 pl-3">{cleanError(span.error)}</Mono>
            )}
          </div>
        ))}
      </div>
    </Section>
  );
}

function DetailHeader({ row, traceId }: { row: OutlineRow; traceId: string }) {
  const span = row.span;
  const stepSpan = row.kind === "input" || row.kind === "output" ? undefined : span;
  const facts = [
    span && row.kind !== "input" && row.kind !== "output" ? span.agent : null,
    row.durationMs != null ? fmtMs(row.durationMs) : null,
    span?.type === "llm" && span.litellm ? fmtCost(span.litellm.spend) : null,
  ].filter(Boolean);
  return (
    <div className="mb-4 flex items-start gap-3">
      <h3 className="min-w-0 flex-1 text-[15px] font-semibold leading-snug break-words">
        {row.error && (
          <span aria-hidden className="mr-2 inline-block size-1.5 -translate-y-0.5 rounded-full bg-destructive" />
        )}
        {row.label}
        {facts.length > 0 && (
          <span className="font-normal text-muted-foreground tabular-nums"> · {facts.join(" · ")}</span>
        )}
      </h3>
      {stepSpan && (
        <Button
          variant="ghost"
          size="xs"
          className="text-muted-foreground"
          onClick={() =>
            void copyToClipboard(copyStepText(agentTraceMarkdownUrl(traceId, stepSpan.span_id)), "Copied step for agent")
          }
        >
          <Copy /> Copy step
        </Button>
      )}
    </div>
  );
}

/** Right pane: the selected outline row, as Content (what happened) and Request (LiteLLM / OTEL facts). */
export function SpanDetail({ accessToken, traceId, row, onOpenRequestLog }: SpanDetailProps) {
  const isSpanRow = row.kind === "span" || row.kind === "input" || row.kind === "output";
  const detailQuery = useSpanDetail(accessToken, traceId, isSpanRow ? row.span : undefined);
  const detail = detailQuery.data;
  const span = row.span;
  const error = row.kind === "span" && span?.status === "error" ? cleanError(span.error) || "Failed" : "";
  const showRequestTab = row.kind === "span" && span !== undefined;

  const content = (
    <>
      {error && <ErrorText error={error} />}
      {!isSpanRow && <GroupContent row={row} />}
      {isSpanRow && detailQuery.isLoading && <div className="text-sm text-muted-foreground">Loading…</div>}
      {isSpanRow && detailQuery.isError && (
        <div className="text-sm text-muted-foreground">Could not load this step: {detailQuery.error.message}</div>
      )}
      {isSpanRow && detail && <SpanContent row={row} detail={detail} />}
      {isSpanRow && !span && <div className="text-sm text-muted-foreground">No root span in this trace.</div>}
    </>
  );

  return (
    <div className="min-h-0 overflow-auto px-6 pt-5 pb-12" data-testid="span-detail">
      <DetailHeader row={row} traceId={traceId} />
      {showRequestTab ? (
        <Tabs defaultValue="content" className="gap-5">
          <TabsList variant="line" className="h-7 gap-4 p-0">
            <TabsTrigger value="content" className="px-0 text-[13px]">
              Content
            </TabsTrigger>
            <TabsTrigger value="request" className="px-0 text-[13px]">
              Request
            </TabsTrigger>
          </TabsList>
          <TabsContent value="content">{content}</TabsContent>
          <TabsContent value="request">
            {span.litellm ? (
              <RequestFacts request={span.litellm} onOpenRequestLog={onOpenRequestLog} />
            ) : (
              detail && <Attributes attributes={detail.attributes} />
            )}
          </TabsContent>
        </Tabs>
      ) : (
        content
      )}
    </div>
  );
}

