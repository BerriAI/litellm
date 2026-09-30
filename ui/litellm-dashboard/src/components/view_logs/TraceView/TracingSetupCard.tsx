"use client";

import { Copy } from "lucide-react";

import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/utils/dataUtils";

export const TRACING_SETUP_SNIPPET = `# proxy config
general_settings:
  tracing:
    store: clickhouse   # + CLICKHOUSE_URL env var
# your agent app
export LANGSMITH_TRACING=true LANGSMITH_TRACING_MODE=otel
export OTEL_EXPORTER_OTLP_ENDPOINT=<proxy url>
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <litellm key>"`;

/** Shown when GET /v1/traces answers 501: one sentence and the snippet. */
export function TracingSetupCard({ detail }: { detail?: string }) {
  return (
    <div data-testid="tracing-setup-card" className="text-[13px]">
      <div className="font-medium">Agent tracing is not enabled</div>
      <p className="mt-1 text-muted-foreground">
        Point your agent&apos;s OpenTelemetry exporter at this proxy to see each run next to its LLM requests.
      </p>
      <div className="relative mt-3">
        <pre className="overflow-x-auto rounded-md bg-muted px-3 py-2.5 pr-10 font-mono text-xs leading-relaxed">
          {TRACING_SETUP_SNIPPET}
        </pre>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="Copy setup snippet"
          className="absolute top-1.5 right-1.5 text-muted-foreground"
          onClick={() => void copyToClipboard(TRACING_SETUP_SNIPPET)}
        >
          <Copy />
        </Button>
      </div>
      {detail && detail !== "Agent tracing is not enabled" && (
        <p className="mt-2 text-xs text-muted-foreground">{detail}</p>
      )}
    </div>
  );
}
