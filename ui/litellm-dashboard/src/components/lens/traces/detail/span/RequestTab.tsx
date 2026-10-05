"use client";

import { ArrowUpRight } from "lucide-react";
import { useState } from "react";

import CopyButton from "@/components/shared/CopyButton";
import { Button } from "@/components/ui/button";

import { LogDetailsDrawer } from "../../../../logs/detail";
import { fieldEntries } from "../content/payload";
import { formatCost } from "../../list/AgentTracesTable";
import type { Span } from "../../types";
import { fmtMs, fmtTok } from "../../utils";
import { useSpanRequestLog } from "../useSpanRequestLog";
import { FieldTree } from "../content/FieldTree";
import { DetailGroup } from "./DetailGroup";

interface RequestTabProps {
  span: Span;
  accessToken: string;
  /** The run's start, so the request log is looked up at the span's time, whatever range the tabs show. */
  traceStartMs: number;
}

/** Usage and the linked LiteLLM request for an LLM span; "Open request log" opens the request drawer over the run. */
export function RequestTab({ span, accessToken, traceStartMs }: RequestTabProps) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const logQuery = useSpanRequestLog(
    accessToken,
    span.litellm_request_id,
    traceStartMs + span.start_offset_ms,
    drawerOpen,
  );
  const logNotFound = drawerOpen && logQuery.isSuccess && logQuery.data === null;

  if (span.type !== "llm") {
    return <div className="py-16 text-center text-sm text-muted-foreground">This span is not a model request.</div>;
  }

  const usage = fieldEntries([
    ["model", span.model ?? "—"],
    ["cost", span.spend == null ? "—" : formatCost(span.spend)],
    ["input_tokens", fmtTok(span.input_tokens)],
    ["output_tokens", fmtTok(span.output_tokens)],
    ["total_tokens", fmtTok(span.input_tokens + span.output_tokens)],
    ["latency", fmtMs(span.duration_ms)],
  ]);

  return (
    <div className="flex flex-col gap-5 px-4 pt-3 pb-5">
      <DetailGroup title="Usage">
        <FieldTree entries={usage} />
      </DetailGroup>
      <DetailGroup title="LiteLLM request">
        {span.litellm_request_id ? (
          <div className="flex flex-col gap-2 py-1.5">
            <div className="flex min-w-0 items-center gap-1">
              <div className="min-w-0 flex-1">
                <FieldTree entries={fieldEntries([["request_id", span.litellm_request_id]])} mono />
              </div>
              <CopyButton variant="action" value={span.litellm_request_id} label="Copy request ID" iconOnly />
            </div>
            <Button variant="outline" size="xs" className="self-start" onClick={() => setDrawerOpen(true)}>
              Open request log <ArrowUpRight className="size-3.5" />
            </Button>
            {logNotFound && <p className="text-sm text-muted-foreground">No request log found for this call.</p>}
          </div>
        ) : (
          <p className="py-1.5 text-sm text-muted-foreground">Not linked to a LiteLLM request</p>
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
