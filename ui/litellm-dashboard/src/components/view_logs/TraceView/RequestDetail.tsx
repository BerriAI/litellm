"use client";

import { ArrowUpRight } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";

import { LogDetailsDrawer } from "../LogDetailsDrawer";
import { CopyButton } from "./CopyButton";
import { formatCost } from "./AgentTracesTable";
import type { Span } from "./traceTypes";
import { fmtMs, fmtTok } from "./traceUtils";
import { useSpanRequestLog } from "./useSpanRequestLog";

interface RequestDetailProps {
  span: Span;
  accessToken: string;
  /** The run's start, so the request log is looked up at the span's time, whatever range the tabs show. */
  traceStartMs: number;
}

/** Request tab for LLM spans; "Open request log" opens the LiteLLM request drawer over the run. */
export function RequestDetail({ span, accessToken, traceStartMs }: RequestDetailProps) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const spanStartMs = traceStartMs + span.start_offset_ms;
  const logQuery = useSpanRequestLog(accessToken, span.litellm_request_id, spanStartMs, drawerOpen);
  const lookupDone = drawerOpen && logQuery.isSuccess;
  const logNotFound = lookupDone && logQuery.data === null;

  if (span.type !== "llm") {
    return (
      <div className="py-16 text-center font-mono text-[11px] text-muted-foreground">
        This span is not a model request.
      </div>
    );
  }

  const rows: [string, string][] = [
    ["Model", span.model ?? "—"],
    ["Cost", span.spend == null ? "—" : formatCost(span.spend)],
    ["Input tokens", fmtTok(span.input_tokens)],
    ["Output tokens", fmtTok(span.output_tokens)],
    ["Total tokens", fmtTok(span.input_tokens + span.output_tokens)],
    ["Latency", fmtMs(span.duration_ms)],
  ];

  return (
    <div className="p-3">
      <dl className="overflow-hidden rounded border border-border bg-card">
        {rows.map(([label, value]) => (
          <div key={label} className="grid min-h-8 grid-cols-[128px_1fr] border-b border-border last:border-0">
            <dt className="flex items-center bg-muted/40 px-2.5 font-mono text-[10px] text-muted-foreground">
              {label}
            </dt>
            <dd className="flex min-w-0 items-center px-2.5 font-mono text-[11px] tabular-nums text-foreground">
              {value}
            </dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 rounded border border-border bg-muted/40 p-3">
        <div className="flex items-center font-mono text-[9px] tracking-[0.1em] text-muted-foreground uppercase">
          Request ID
          {span.litellm_request_id && (
            <CopyButton value={span.litellm_request_id} label="Copy request ID" iconOnly className="ml-auto" />
          )}
        </div>
        <div className="mt-1.5 font-mono text-[11px] break-all text-foreground">
          {span.litellm_request_id ?? "Not linked to a LiteLLM request"}
        </div>
      </div>
      {span.litellm_request_id && (
        <Button
          variant="outline"
          size="sm"
          className="mt-3 w-full gap-1.5 font-mono text-[11px]"
          onClick={() => setDrawerOpen(true)}
        >
          Open request log <ArrowUpRight className="size-3" />
        </Button>
      )}
      {logNotFound && (
        <p className="mt-2 font-mono text-[11px] text-muted-foreground">No request log found for this call.</p>
      )}
      <LogDetailsDrawer
        open={drawerOpen && Boolean(logQuery.data)}
        onClose={() => setDrawerOpen(false)}
        logEntry={logQuery.data ?? null}
        accessToken={accessToken}
      />
    </div>
  );
}
