"use client";

import { ArrowUpRight } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";

import { LogDetailsDrawer } from "../../../logs/detail";
import { formatCost } from "../list/AgentTracesTable";
import { DetailGroup } from "./AttributesDetail";
import { CopyButton } from "../ui/CopyButton";
import { type KeyValue, KeyValueRows } from "./KeyValueRows";
import type { Span } from "../types";
import { fmtMs, fmtTok } from "../utils";
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
    return <div className="py-16 text-center text-sm text-trace-duration">This span is not a model request.</div>;
  }

  const usage: KeyValue[] = [
    ["model", span.model ?? "—"],
    ["cost", span.spend == null ? "—" : formatCost(span.spend)],
    ["input_tokens", fmtTok(span.input_tokens)],
    ["output_tokens", fmtTok(span.output_tokens)],
    ["total_tokens", fmtTok(span.input_tokens + span.output_tokens)],
    ["latency", fmtMs(span.duration_ms)],
  ];

  return (
    <div className="flex flex-col gap-4 px-7 pt-1 pb-4">
      <DetailGroup title="Usage">
        <KeyValueRows entries={usage} />
      </DetailGroup>
      <DetailGroup title="LiteLLM request">
        {span.litellm_request_id ? (
          <div className="flex flex-col gap-2.5">
            <div className="flex min-w-0 items-center gap-1">
              <KeyValueRows entries={[["request_id", span.litellm_request_id]]} mono className="min-w-0 flex-1" />
              <CopyButton value={span.litellm_request_id} label="Copy request ID" iconOnly />
            </div>
            <Button
              variant="outline"
              size="xs"
              className="self-start border-trace-border text-sm text-trace-text-2 shadow-trace-xs"
              onClick={() => setDrawerOpen(true)}
            >
              Open request log <ArrowUpRight className="size-3.5" />
            </Button>
            {logNotFound && <p className="text-sm text-trace-duration">No request log found for this call.</p>}
          </div>
        ) : (
          <p className="text-sm text-trace-duration">Not linked to a LiteLLM request</p>
        )}
      </DetailGroup>
      <LogDetailsDrawer
        open={drawerOpen && Boolean(logQuery.data)}
        onClose={() => setDrawerOpen(false)}
        logEntry={logQuery.data ?? null}
        accessToken={accessToken}
      />
    </div>
  );
}
