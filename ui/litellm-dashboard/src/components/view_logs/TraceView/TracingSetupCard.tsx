"use client";

import { Info } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export const TRACING_SETUP_SNIPPET = `# proxy config
general_settings:
  tracing:
    store: clickhouse   # + CLICKHOUSE_URL env var
# your agent app
export LANGSMITH_TRACING=true LANGSMITH_TRACING_MODE=otel
export OTEL_EXPORTER_OTLP_ENDPOINT=<proxy url>
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <litellm key>"`;

/** Shown when GET /v1/traces answers 501 (agent tracing not configured on this proxy). */
export function TracingSetupCard({ detail }: { detail?: string }) {
  return (
    <Alert className="mb-4" data-testid="tracing-setup-card">
      <Info />
      <AlertTitle>Agent tracing is not enabled</AlertTitle>
      <AlertDescription>
        <p>
          Point your agent&apos;s OpenTelemetry exporter at this proxy to see agent runs here, joined to their LLM
          request logs.{detail ? ` (${detail})` : ""}
        </p>
        <pre className="mt-2 w-full overflow-x-auto rounded-md border bg-muted px-3 py-2 font-mono text-xs text-foreground">
          {TRACING_SETUP_SNIPPET}
        </pre>
      </AlertDescription>
    </Alert>
  );
}
