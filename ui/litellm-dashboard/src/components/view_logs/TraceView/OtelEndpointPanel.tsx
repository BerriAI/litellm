"use client";

import { BookOpen } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/cva.config";

import { getProxyBaseUrl } from "../../networking";
import { relativeTime } from "./AgentTracesTable";
import { CopyButton } from "./CopyButton";
import { tracingEnvSnippet } from "./TracingSetupCard";

interface EndpointRow {
  label: string;
  value: string;
  copy?: boolean;
}

export const proxyBaseUrl = (): string => getProxyBaseUrl().replace(/\/$/, "");

export const proxyHost = (proxyUrl: string): string => {
  try {
    return new URL(proxyUrl).host || proxyUrl;
  } catch {
    return proxyUrl;
  }
};

/** Facts about the OTLP receiver: the SDK endpoint is the proxy base URL, it appends /v1/traces itself. */
export const endpointRows = (proxyUrl: string): EndpointRow[] => [
  { label: "OTLP endpoint", value: proxyUrl, copy: true },
  { label: "Receives", value: `POST ${proxyUrl}/v1/traces` },
  { label: "Protocol", value: "OTLP/HTTP, protobuf or JSON" },
  { label: "Auth", value: "Virtual key in the Authorization header. Runs are scoped to that key and its team." },
  { label: "Read API", value: `GET ${proxyUrl}/v1/traces · /v1/traces/{trace_id} · ?format=md` },
];

interface OtelEndpointPanelProps {
  /** Start time of the newest loaded run, or null when the range has none. */
  newestRunStart: string | null;
  onOpenGuide: () => void;
}

function StatusDot({ active }: { active: boolean }) {
  return (
    <span aria-hidden className={cn("size-1.5 shrink-0 rounded-full", active ? "bg-info" : "bg-muted-foreground/50")} />
  );
}

function Row({ row }: { row: EndpointRow }) {
  return (
    <div className="grid grid-cols-[88px_minmax(0,1fr)] items-start gap-2">
      <dt className="pt-0.5 font-mono text-[10px] tracking-wide text-muted-foreground uppercase">{row.label}</dt>
      <dd className="flex min-w-0 items-start gap-1.5 text-[12px] text-foreground">
        <span className="min-w-0 font-mono break-all">{row.value}</span>
        {row.copy && <CopyButton value={row.value} label={`Copy ${row.label}`} iconOnly />}
      </dd>
    </div>
  );
}

/** Always-visible OTLP pill in the runs toolbar; opens the endpoint, env vars and receive status. */
export function OtelEndpointPanel({ newestRunStart, onOpenGuide }: OtelEndpointPanelProps) {
  const proxyUrl = proxyBaseUrl();
  const receiving = newestRunStart !== null;
  const env = tracingEnvSnippet(proxyUrl);
  const status = receiving ? `Receiving traces, newest run ${relativeTime(newestRunStart)}` : "No runs in this range";

  return (
    <Popover>
      <PopoverTrigger
        aria-label="OpenTelemetry endpoint"
        className="inline-flex h-7 shrink-0 items-center gap-1.5 rounded-md border border-border px-2 font-mono text-[11px] text-muted-foreground hover:text-foreground"
      >
        <StatusDot active={receiving} />
        <span className="text-foreground">OTLP</span>
        <span>{proxyHost(proxyUrl)}</span>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[420px] max-w-[calc(100vw-32px)] gap-3">
        <div className="flex flex-col gap-0.5">
          <h2 className="text-[13px] font-medium text-foreground">OpenTelemetry endpoint</h2>
          <p className="text-[12px] text-muted-foreground">
            Point any OpenTelemetry SDK here. LiteLLM stores each run as an agent trace.
          </p>
        </div>
        <dl className="flex flex-col gap-2">
          {endpointRows(proxyUrl).map((row) => (
            <Row key={row.label} row={row} />
          ))}
        </dl>
        <div className="overflow-hidden rounded-md border border-border bg-muted/30">
          <div className="flex h-7 items-center border-b border-border px-2 text-[11px] text-muted-foreground">
            Shell
            <span className="ml-auto">
              <CopyButton value={env} label="Copy env vars" iconOnly />
            </span>
          </div>
          <pre className="overflow-x-auto px-2 py-1.5 font-mono text-[11px] leading-5 text-foreground">
            <code>{env}</code>
          </pre>
        </div>
        <div className="flex items-center gap-2 text-[12px] text-muted-foreground" data-testid="otel-status">
          <StatusDot active={receiving} />
          {status}
          <Button variant="outline" size="xs" className="ml-auto gap-1.5 text-[11px]" onClick={onOpenGuide}>
            <BookOpen className="size-3" /> Full setup guide
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
