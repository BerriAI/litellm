"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

import { useTracesApi } from "../../api";
import type { Span, SpanDetail, UIContent } from "../../types";
import { parseMessages, prettyPayload } from "../../utils";
import { Payload, TextBody } from "./PayloadBody";
import { ToolArguments } from "./ToolContent";
import { payloadView, toolInput } from "./payload";
import { Section } from "./Section";
import { ErrorBlock, StoredDiagnostic } from "./SpanError";

const STATUS_TEXT = "px-4 py-2 text-sm text-muted-foreground";
const COLLAPSE_INPUT_ABOVE = 8;

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

const messageCount = (raw: string, content: UIContent | undefined): number | undefined =>
  content?.kind === "messages" ? content.messages.length : parseMessages(raw)?.length;

function PayloadSection({
  title,
  raw,
  content,
  span,
  role,
}: {
  title: string;
  raw: string;
  content: UIContent | undefined;
  span: Span;
  role: "input" | "output";
}) {
  const view = payloadView(
    raw,
    content,
    span.type === "tool" && role === "output",
    span.type !== "tool" && role === "output",
  );
  const count = role === "input" ? messageCount(raw, content) : undefined;
  return (
    <Section
      key={`${span.span_id}:${count}`}
      title={title}
      count={count}
      defaultOpen={!count || count <= COLLAPSE_INPUT_ABOVE}
    >
      {(mode) => {
        if (mode === "Raw") return <TextBody text={prettyPayload(raw)} format="code" />;
        if (span.type === "tool" && role === "input") return <ToolArguments args={toolInput(raw, content)} />;
        return <Payload view={view} name={span.name} failed={span.status === "error"} />;
      }}
    </Section>
  );
}

interface ContentTabProps {
  accessToken: string;
  traceId: string;
  traceRef?: string;
  span: Span;
}

/** The error first (if any), then collapsible Input and Output. */
export function ContentTab({ accessToken, traceId, traceRef, span }: ContentTabProps) {
  const detailQuery = useSpanDetail(accessToken, traceId, span.span_id, traceRef);
  const detail = detailQuery.data;
  const empty = detail && !detail.input && !detail.output;
  return (
    <div className="flex flex-col pb-4">
      {(span.status === "error" || span.error) && (
        <div className="flex flex-col gap-2 px-4 pt-3 pb-1">
          <ErrorBlock span={span} />
          {span.error && (
            <StoredDiagnostic
              key={`${traceId}:${traceRef}:${span.span_id}`}
              accessToken={accessToken}
              traceId={traceId}
              traceRef={traceRef}
              span={span}
            />
          )}
        </div>
      )}
      {detailQuery.isLoading && <div className={STATUS_TEXT}>Loading span…</div>}
      {detailQuery.isError && <div className={STATUS_TEXT}>Could not load span: {detailQuery.error.message}</div>}
      {detail?.input ? (
        <PayloadSection title="Input" raw={detail.input} content={detail.input_ui} span={span} role="input" />
      ) : null}
      {detail?.output ? (
        <PayloadSection title="Output" raw={detail.output} content={detail.output_ui} span={span} role="output" />
      ) : null}
      {empty && span.status !== "error" && (
        <div className="py-12 text-center text-sm text-muted-foreground">No content recorded for this span.</div>
      )}
    </div>
  );
}
