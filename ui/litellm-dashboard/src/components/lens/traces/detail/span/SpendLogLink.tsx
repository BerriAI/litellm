"use client";

import { ArrowUpRight } from "lucide-react";
import { useState } from "react";

import { LogDetailsDrawer } from "../../../../logs/detail";
import { formatCost } from "../../list/AgentTracesTable";
import type { Span } from "../../types";
import { useSpanRequestLog } from "../useSpanRequestLog";

const UNMATCHED_REASON: Record<Exclude<NonNullable<Span["spend_match"]>, "matched">, string> = {
  no_call_id: "This step records no gen_ai.response.id or litellm.call_id",
  no_spend_log: "No spend log carries this step's id",
  ambiguous: "More than one spend log carries this step's id",
};

export const unmatchedReason = (span: Span): string | null =>
  span.spend_match && span.spend_match !== "matched" ? UNMATCHED_REASON[span.spend_match] : null;

const shortId = (id: string): string => (id.length > 12 ? `${id.slice(0, 10)}…` : id);

/** The spend log a model call was priced from, opened in the request log drawer over the run. */
export function SpendLogLink({
  span,
  accessToken,
  traceStartMs,
}: {
  span: Span;
  accessToken: string;
  traceStartMs: number;
}) {
  const [open, setOpen] = useState(false);
  const requestId = span.spend_log_request_id ?? null;
  const logQuery = useSpanRequestLog(accessToken, requestId, traceStartMs + span.start_offset_ms, open);
  const reason = unmatchedReason(span);

  if (requestId == null) {
    if (!reason) return null;
    return (
      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground" title={reason}>
        Cost <span className="font-medium text-foreground">not matched</span>
      </span>
    );
  }

  const missing = open && logQuery.isSuccess && logQuery.data === null;
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={requestId}
        aria-label={`Open spend log ${requestId}`}
        className="inline-flex h-6 items-center gap-1 rounded-md px-1.5 text-xs text-muted-foreground tabular-nums transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring active:scale-[0.97] motion-reduce:transition-none"
      >
        <span>Spend log</span>
        <span aria-hidden>·</span>
        <span className="font-medium text-foreground">{shortId(requestId)}</span>
        {span.spend != null && (
          <>
            <span aria-hidden>·</span>
            <span className="font-medium text-foreground">{formatCost(span.spend)}</span>
          </>
        )}
        <ArrowUpRight className="size-3.5" />
      </button>
      {missing && <span className="text-xs text-muted-foreground">Spend log not visible in this time range</span>}
      <LogDetailsDrawer
        open={open && Boolean(logQuery.data)}
        onClose={() => setOpen(false)}
        logEntry={logQuery.data ?? null}
        accessToken={accessToken}
      />
    </>
  );
}
