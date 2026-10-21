"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { TriangleAlert } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";

import { useTracesApi } from "../../api";
import {
  classifyTraceReadFailure,
  isRetryableTraceRead,
  traceReadRetry,
  traceReadRetryDelay,
} from "../../list/traceReadFailure";
import type { ErrorSource } from "../../tree";
import type { Span, SpanErrorPage } from "../../types";
import { errorSource } from "../../utils";
import { TextBody } from "./PayloadBody";

const ERROR_SOURCE_LABEL: Record<ErrorSource, string> = { tool: "Tool", model: "Model", litellm: "LiteLLM" };
const TRACEBACK_MARKER = "Traceback (most recent call last):";

/** Exporters record `repr(exc)` + traceback with no separator; keep the exception line. */
export const errorHeadline = (error: string): string =>
  (error.split(TRACEBACK_MARKER, 1)[0].split("\n")[0] ?? "").trim() || error.trim();

/** `ValueError('x not found')` → "ValueError"; plain text → "error". */
const errorReason = (headline: string): string => /^([A-Za-z_][\w.]*)\(/.exec(headline)?.[1] ?? "error";

export function ErrorBlock({ span }: { span: Span }) {
  const source = errorSource(span);
  if (!source) return null;
  const headline = errorHeadline(span.error ?? "") || "Span reported an error status.";
  return (
    <section aria-label="Error" className="rounded-lg border border-destructive/40 bg-destructive/5 px-3.5 py-3">
      <div className="flex items-center gap-2 text-sm font-semibold text-destructive">
        <TriangleAlert className="size-4" />
        {ERROR_SOURCE_LABEL[source]} · {errorReason(headline)}
      </div>
      <pre className="mt-2 font-mono text-xs leading-relaxed break-words whitespace-pre-wrap text-foreground">
        {headline}
      </pre>
    </section>
  );
}

interface DiagnosticProps {
  accessToken: string;
  traceId: string;
  traceRef?: string;
  span: Span;
}

/** The full stored error, fetched only on request and paged section by section. */
export function StoredDiagnostic({ accessToken, traceId, traceRef, span }: DiagnosticProps) {
  const traces = useTracesApi(accessToken);
  const [opened, setOpened] = useState(false);
  const [cursor, setCursor] = useState<string | null>(null);
  const queryOptions: UseQueryOptions<SpanErrorPage, Error> = {
    queryKey: ["agentTraceSpanError", traceId, traceRef, span.span_id, accessToken, cursor],
    queryFn: () => traces.spanError(traceId, span.span_id, { traceRef, cursor }),
    enabled: opened,
    staleTime: Infinity,
    gcTime: 0,
    retry: traceReadRetry,
    retryDelay: traceReadRetryDelay,
  };
  const query = useQuery(queryOptions);
  const failure = query.error ? classifyTraceReadFailure(query.error) : null;
  return (
    <section aria-label="Stored diagnostic" className="flex flex-col items-start gap-2">
      {span.error_truncated && <p className="text-xs text-muted-foreground">Error preview truncated</p>}
      {!opened && (
        <Button variant="outline" size="xs" onClick={() => setOpened(true)}>
          View stored diagnostic
        </Button>
      )}
      {opened && query.isPending && (
        <p role="status" className="text-xs text-muted-foreground">
          Loading diagnostic…
        </p>
      )}
      {failure && (
        <div role="alert" className="flex items-center gap-2 text-xs">
          Could not load diagnostic: {failure.message}
          {isRetryableTraceRead(failure) || !cursor ? (
            <Button variant="outline" size="xs" onClick={() => query.refetch()}>
              Retry
            </Button>
          ) : (
            <Button variant="outline" size="xs" onClick={() => setCursor(null)}>
              Back to beginning
            </Button>
          )}
        </div>
      )}
      {opened && query.data && (
        <>
          <div className="w-full">
            <TextBody text={query.data.message} format="code" />
          </div>
          <p className="text-xs text-muted-foreground">
            {cursor ? "Continuation" : "Beginning"} of stored diagnostic ({query.data.total_chars.toLocaleString()}{" "}
            characters)
          </p>
          <div className="flex gap-2">
            {query.data.next_cursor && (
              <Button variant="outline" size="xs" onClick={() => setCursor(query.data.next_cursor)}>
                Next section
              </Button>
            )}
            {cursor && (
              <Button variant="outline" size="xs" onClick={() => setCursor(null)}>
                Back to beginning
              </Button>
            )}
          </div>
        </>
      )}
    </section>
  );
}
