"use client";

import { ArrowRight, ArrowUpRight, Check, Copy, KeyRound, Loader2, Send } from "lucide-react";
import { useState } from "react";
import { useTimeout } from "usehooks-ts";

import { cn } from "@/lib/cva.config";
import { LensPreviewButton } from "./LensPreviewButton";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { copyToClipboard } from "@/utils/dataUtils";

import anthropicLogo from "../../../../public/assets/logos/anthropic.svg";
import openaiLogo from "../../../../public/assets/logos/openai_small.svg";
import otelLogo from "../../../../public/assets/logos/opentelemetry.svg";
import { agentTraceCall, apiClient, getProxyBaseUrl, sendOtlpTraceCall } from "../../networking";
import { ActiveDot } from "./ActiveDot";
import { sampleTraceExport } from "./sampleTrace";
import { FRAMEWORKS, frameworkSnippet, type FrameworkGuide } from "./tracingSetupGuides";
import type { TraceSummary } from "./traceTypes";

const COPIED_RESET_MS = 1500;
const DOCS_URL = "https://docs.litellm.ai/docs/proxy/lens";
const EXAMPLE_MODEL = "openai/gpt-6-sol";
const SAMPLE_TRACE_POLL_MS = 1000;
export const TRACING_KEY_REQUEST = {
  key_alias: "Agent tracing",
  allowed_routes: ["/v1/traces"],
  metadata: { purpose: "agent_tracing" },
} as const;
const SAMPLE_TRACE_POLL_ATTEMPTS = 15;

type Installer = "pip" | "uv";
type CodingAgent = "Claude Code" | "Codex";

const PY_INSTALL: Record<Installer, (packages: string) => string> = {
  pip: (packages) => `pip install ${packages}`,
  uv: (packages) => `uv add ${packages}`,
};

export const tracingEnvSnippet = (proxyUrl: string, tracingKey: string | null = null): string =>
  [
    ...(tracingKey ? [`export LITELLM_TRACING_KEY=${tracingKey}`] : []),
    `export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${proxyUrl}/v1/traces"`,
    `export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer $${tracingKey ? "LITELLM_TRACING_KEY" : "LITELLM_API_KEY"}"`,
    'export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"',
    'export OTEL_METRICS_EXPORTER="none"',
    'export OTEL_LOGS_EXPORTER="none"',
  ].join("\n");

export const codingAgentPrompt = (proxyUrl: string, guide: FrameworkGuide, model: string): string =>
  [
    `Send this ${guide.label} project's OpenTelemetry traces to LiteLLM.`,
    "Keep the existing model configuration, authentication, and application behavior. Never hardcode a key; read it from LITELLM_API_KEY.",
    "Set the trace destination wherever this project loads environment variables:",
    tracingEnvSnippet(proxyUrl),
    guide.install ?? `Install and enable the ${guide.plugin?.label}: ${guide.plugin?.url}`,
    guide.plugin?.instruction ??
      "Initialize OpenTelemetry before creating the agent. If the app already configures a tracer provider, keep it and point its exporter at the destination above instead.",
    "Adapt this example to the existing application, replacing research_agent with the agent's name:",
    frameworkSnippet(guide, proxyUrl, model),
    guide.note ?? "",
    "Run the agent once and confirm its named run appears in Lens > Traces.",
  ]
    .filter(Boolean)
    .join("\n\n");

const shellQuote = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`;

export const codingAgentCommand = (agent: CodingAgent, prompt: string): string =>
  `${agent === "Claude Code" ? "claude" : "codex"} ${shellQuote(prompt)}`;

export const maskSecret = (secret: string): string =>
  secret.length > 10 ? `${secret.slice(0, 5)}${"•".repeat(16)}${secret.slice(-4)}` : "•".repeat(secret.length);

export const otlpEndpoints = (proxyUrl: string): readonly (readonly [string, string, boolean])[] => [
  ["Traces endpoint", `${proxyUrl}/v1/traces`, true],
  ["Auth header", "Authorization: Bearer <LiteLLM virtual key>", true],
  ["Protocol", "OTLP/HTTP (protobuf or JSON)", false],
];

export const PROXY_CONFIG_SNIPPET = [
  "general_settings:",
  "  tracing:",
  "    store:",
  "      type: clickhouse",
  "      url: os.environ/CLICKHOUSE_URL",
  "      retention_days: 14",
].join("\n");

function CodeBlock({
  code,
  display = code,
  tabs,
  wrap = false,
}: {
  code: string;
  display?: string;
  tabs?: React.ReactNode;
  wrap?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  useTimeout(() => setCopied(false), copied ? COPIED_RESET_MS : null);
  const copy = async () => setCopied(await copyToClipboard(code));
  return (
    <div className="overflow-hidden rounded-md border border-border bg-muted/30">
      <div className="flex h-9 items-center border-b border-border px-3">
        {tabs}
        <button
          type="button"
          onClick={() => void copy()}
          aria-label="Copy"
          className="ml-auto text-muted-foreground hover:text-foreground"
        >
          {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
        </button>
      </div>
      <pre
        className={cn(
          "m-0 px-4 py-3 font-mono text-xs leading-5 text-foreground",
          wrap ? "whitespace-pre-wrap break-words" : "overflow-x-auto",
        )}
      >
        <code>{display}</code>
      </pre>
    </div>
  );
}

function FileLabel({ children }: { children: React.ReactNode }) {
  return <span className="text-xs text-foreground">{children}</span>;
}

function LineTabs<T extends string>({
  value,
  options,
  onChange,
  logos,
}: {
  value: T;
  options: T[];
  onChange: (v: T) => void;
  logos?: Partial<Record<T, string>>;
}) {
  return (
    <div role="tablist" className="flex h-9 items-end gap-3">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          role="tab"
          aria-selected={value === option}
          onClick={() => onChange(option)}
          className={cn(
            "-mb-px inline-flex h-9 items-center gap-1.5 border-b-2 text-xs",
            value === option
              ? "border-foreground text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground",
          )}
        >
          {logos?.[option] && <img src={logos[option]} alt="" className="size-3.5" />}
          {option}
        </button>
      ))}
    </div>
  );
}

function Step({ title, children }: { title: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="mt-6">
      <h3 className="mb-3 text-sm font-medium text-foreground">{title}</h3>
      {children}
    </section>
  );
}

type SendState =
  | { kind: "idle" }
  | { kind: "sending" }
  | { kind: "waiting" }
  | { kind: "ready"; trace: TraceSummary }
  | { kind: "failed"; message: string };

const SEND_LABEL: Record<Exclude<SendState["kind"], "ready">, string> = {
  idle: "Send a test trace",
  sending: "Sending…",
  waiting: "Waiting for it to arrive…",
  failed: "Send a test trace",
};

const delay = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

async function waitForTrace(accessToken: string, traceId: string): Promise<TraceSummary | null> {
  for (let attempt = 0; attempt < SAMPLE_TRACE_POLL_ATTEMPTS; attempt++) {
    try {
      return (await agentTraceCall(accessToken, traceId)).summary;
    } catch {
      await delay(SAMPLE_TRACE_POLL_MS);
    }
  }
  return null;
}

function SendTestTrace({
  accessToken,
  onOpenTrace,
}: {
  accessToken: string;
  onOpenTrace: (trace: TraceSummary) => void;
}) {
  const [state, setState] = useState<SendState>({ kind: "idle" });
  const send = async () => {
    setState({ kind: "sending" });
    const sample = sampleTraceExport(Date.now());
    try {
      await sendOtlpTraceCall(accessToken, sample.body);
    } catch {
      setState({ kind: "failed", message: "Could not send the test trace." });
      return;
    }
    setState({ kind: "waiting" });
    const trace = await waitForTrace(accessToken, sample.traceId);
    setState(
      trace
        ? { kind: "ready", trace }
        : { kind: "failed", message: "Sent, but it has not shown up yet. Check again in a moment." },
    );
  };
  if (state.kind === "ready") {
    return (
      <div className="flex flex-wrap items-center gap-3">
        <span role="status" className="inline-flex items-center gap-1 text-sm text-emerald-700 dark:text-emerald-400">
          <Check aria-hidden="true" className="size-4" /> Test trace received
        </span>
        <Button onClick={() => onOpenTrace(state.trace)}>
          View trace <ArrowRight aria-hidden="true" className="size-4" />
        </Button>
      </div>
    );
  }
  const busy = state.kind === "sending" || state.kind === "waiting";
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Button variant="outline" onClick={() => void send()} disabled={busy}>
        {busy ? (
          <Loader2 aria-hidden="true" className="size-4 animate-spin" />
        ) : (
          <Send aria-hidden="true" className="size-4" />
        )}
        {SEND_LABEL[state.kind]}
      </Button>
      {state.kind === "failed" && <p className="text-sm text-destructive">{state.message}</p>}
    </div>
  );
}

function TraceReceipt({
  connected,
  checked,
  checking,
  onCheck,
}: {
  connected: boolean;
  checked: boolean;
  checking: boolean;
  onCheck: () => void;
}) {
  const missingAfterCheck = checked && !checking && !connected;
  return (
    <section className="mt-6 border-t pt-6" aria-label="Trace connection status">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p role="status" className="text-sm font-medium">
            {connected ? "Traces connected" : "Waiting for your first trace"}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">Run your agent once. Its traces will appear here.</p>
        </div>
        <Button variant="outline" onClick={onCheck} disabled={checking}>
          {checking && <Loader2 aria-hidden="true" className="size-4 animate-spin" />}
          {checking ? "Checking…" : "Check for traces"}
        </Button>
      </div>
      {missingAfterCheck && (
        <p className="mt-3 text-sm leading-6 text-muted-foreground">
          No traces received yet. Check the exporter URL and LiteLLM key in your agent’s environment, then check its
          logs for export errors.
        </p>
      )}
    </section>
  );
}

function TracingKey({
  accessToken,
  tracingKey,
  onCreated,
}: {
  accessToken: string;
  tracingKey: string | null;
  onCreated: (key: string) => void;
}) {
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const create = async () => {
    setCreating(true);
    setError("");
    try {
      const result = await apiClient.post<{ key?: string }>("/key/generate", {
        accessToken,
        body: TRACING_KEY_REQUEST,
      });
      if (!result.key) throw new Error("The proxy did not return the new key");
      onCreated(result.key);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not create a key");
    } finally {
      setCreating(false);
    }
  };
  if (tracingKey) {
    return (
      <div className="space-y-2">
        <CodeBlock code={tracingKey} display={maskSecret(tracingKey)} tabs={<FileLabel>Your tracing key</FileLabel>} />
        <p className="text-sm text-muted-foreground">
          Hidden for safety. Copy copies the full key, and the environment step below includes it. This key can only
          send traces, so your agent still needs its own key for model calls. Manage it under Virtual Keys as
          &quot;Agent tracing&quot;.
        </p>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Button variant="outline" onClick={() => void create()} disabled={creating}>
        {creating ? (
          <Loader2 aria-hidden="true" className="size-4 animate-spin" />
        ) : (
          <KeyRound aria-hidden="true" className="size-4" />
        )}
        Generate tracing key
      </Button>
      <span className="text-sm text-muted-foreground">Or use any existing LiteLLM virtual key.</span>
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}

function EndpointValue({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  useTimeout(() => setCopied(false), copied ? COPIED_RESET_MS : null);
  const copy = async () => setCopied(await copyToClipboard(value));
  return (
    <button
      type="button"
      onClick={() => void copy()}
      aria-label={`Copy ${value}`}
      className="group inline-flex min-w-0 items-center gap-2 text-left font-mono text-xs text-foreground"
    >
      <span className="break-all">{value}</span>
      {copied ? (
        <Check aria-hidden="true" className="size-3.5 shrink-0" />
      ) : (
        <Copy aria-hidden="true" className="size-3.5 shrink-0 text-muted-foreground group-hover:text-foreground" />
      )}
    </button>
  );
}

function Endpoints({ proxyUrl, children }: { proxyUrl: string; children?: React.ReactNode }) {
  return (
    <section className="mt-6" aria-labelledby="otel-endpoints">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 id="otel-endpoints" className="inline-flex items-center gap-2 text-sm font-medium">
          <img src={otelLogo.src} alt="" className="size-5" />
          Connection details
        </h3>
        {children}
      </div>
      <dl className="mt-3 grid gap-x-6 gap-y-3 rounded-md border bg-muted/30 p-4 md:grid-cols-2">
        {otlpEndpoints(proxyUrl).map(([label, value, copyable]) => (
          <div key={label} className={cn("min-w-0", label === "Traces endpoint" && "md:col-span-2")}>
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="mt-0.5">
              {copyable ? <EndpointValue value={value} /> : <span className="text-xs">{value}</span>}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

const CODING_AGENT_LOGOS: Record<CodingAgent, string> = {
  "Claude Code": anthropicLogo.src,
  Codex: openaiLogo.src,
};

function setupTitle(enabled: boolean, connected: boolean) {
  if (!enabled) return "Enable tracing";
  return connected ? "Connect another agent" : "Connect your agent";
}

interface ConnectAgentProps {
  accessToken: string;
  onOpenTrace: (trace: TraceSummary) => void;
  connected: boolean;
  checked: boolean;
  checking: boolean;
  onCheck: () => void;
  readOnly: boolean;
  canMintTracingKey: boolean;
}

function EnableTracing({ checked, checking, onCheck }: { checked: boolean; checking: boolean; onCheck: () => void }) {
  return (
    <>
      <Step title="Proxy configuration">
        <p className="mb-3 text-sm leading-6 text-muted-foreground">
          Set your ClickHouse URL, add this to config.yaml, then restart the proxy. Ask your proxy administrator if you
          don’t manage this deployment.
        </p>
        <CodeBlock code={PROXY_CONFIG_SNIPPET} tabs={<FileLabel>config.yaml</FileLabel>} />
        <a
          className="mt-3 inline-flex items-center gap-1 text-sm underline underline-offset-4"
          href={`${DOCS_URL}#configure-an-existing-proxy`}
          target="_blank"
          rel="noreferrer"
        >
          ClickHouse and proxy setup <ArrowUpRight aria-hidden="true" className="size-3.5" />
        </a>
      </Step>
      {checked && !checking && (
        <p className="mt-4 text-sm text-muted-foreground">
          Tracing is still unavailable. Check that the configuration was applied to this proxy and it has restarted.
        </p>
      )}
      <div className="mt-6 border-t pt-6">
        <Button onClick={onCheck} disabled={checking}>
          {checking && <Loader2 aria-hidden="true" className="size-4 animate-spin" />}
          {checking ? "Checking…" : "Check setup"}
        </Button>
      </div>
    </>
  );
}

function CodingAgentSetup({ proxyUrl, guide, model }: { proxyUrl: string; guide: FrameworkGuide; model: string }) {
  const [codingAgent, setCodingAgent] = useState<CodingAgent>("Claude Code");
  const [copied, setCopied] = useState<string | null>(null);
  const command = codingAgentCommand(codingAgent, codingAgentPrompt(proxyUrl, guide, model));
  useTimeout(() => setCopied(null), copied === null ? null : COPIED_RESET_MS);
  const copy = async () => {
    if (await copyToClipboard(command)) setCopied(command);
  };
  return (
    <section className="mt-6" aria-labelledby="connect-project">
      <h3 id="connect-project" className="text-sm font-medium">
        Connect your project
      </h3>
      <div className="mt-3 overflow-hidden rounded-md border">
        <div className="border-b bg-muted/30 px-4">
          <LineTabs
            value={codingAgent}
            options={["Claude Code", "Codex"]}
            onChange={(value) => {
              setCodingAgent(value);
              setCopied(null);
            }}
            logos={CODING_AGENT_LOGOS}
          />
        </div>
        <div className="p-4">
          <p className="text-sm leading-6 text-muted-foreground">
            Run the setup command in your agent’s project. It uses your <code className="text-xs">LITELLM_API_KEY</code>
            .
          </p>
          <Button className="mt-3" onClick={() => void copy()}>
            {copied === command ? (
              <Check aria-hidden="true" className="size-4" />
            ) : (
              <Copy aria-hidden="true" className="size-4" />
            )}
            {copied === command ? "Copied" : "Copy setup command"}
          </Button>
          <details className="mt-3">
            <summary className="w-fit cursor-pointer text-xs text-muted-foreground">View command</summary>
            <pre className="mt-3 max-h-56 overflow-auto whitespace-pre-wrap rounded-md bg-muted/40 p-3 text-xs leading-5">
              <code>{command}</code>
            </pre>
          </details>
        </div>
      </div>
    </section>
  );
}

function ConnectAgent({
  accessToken,
  onOpenTrace,
  connected,
  checked,
  checking,
  onCheck,
  readOnly,
  canMintTracingKey,
}: ConnectAgentProps) {
  const proxyUrl = getProxyBaseUrl().replace(/\/$/, "");
  const [framework, setFramework] = useState(FRAMEWORKS[0].id);
  const [installer, setInstaller] = useState<Installer>("pip");
  const [tracingKey, setTracingKey] = useState<string | null>(null);
  const guide = FRAMEWORKS.find((f) => f.id === framework) ?? FRAMEWORKS[0];
  const install = guide.install?.startsWith("pip install ")
    ? PY_INSTALL[installer](guide.install.slice("pip install ".length))
    : guide.install;
  const model = EXAMPLE_MODEL;
  const quickstart = frameworkSnippet(guide, proxyUrl, model, !!tracingKey);
  return (
    <>
      <Endpoints proxyUrl={proxyUrl}>
        {!readOnly && <SendTestTrace accessToken={accessToken} onOpenTrace={onOpenTrace} />}
      </Endpoints>
      <div className="mt-6 space-y-2">
        <label id="tracing-framework" className="block text-sm font-medium">
          Your agent framework
        </label>
        <Select
          value={framework}
          onValueChange={(value) => {
            if (value) setFramework(value);
          }}
        >
          <SelectTrigger aria-labelledby="tracing-framework" className="w-full">
            <SelectValue>
              <img src={guide.logo} alt="" className="size-4 shrink-0 object-contain" />
              {guide.label}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {FRAMEWORKS.map((f) => (
              <SelectItem key={f.id} value={f.id}>
                <img src={f.logo} alt="" className="size-4 shrink-0 object-contain" />
                {f.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <CodingAgentSetup proxyUrl={proxyUrl} guide={guide} model={model} />
      <details className="mt-6 border-t pt-6">
        <summary className="w-fit cursor-pointer text-sm font-medium">Set up manually</summary>
        <Step title="Get a LiteLLM key">
          {canMintTracingKey && !readOnly ? (
            <TracingKey accessToken={accessToken} tracingKey={tracingKey} onCreated={setTracingKey} />
          ) : (
            <p className="text-sm text-muted-foreground">
              Use any LiteLLM virtual key you already have, or ask a proxy admin for one.
            </p>
          )}
        </Step>
        {install && (
          <Step title="Install dependencies">
            <CodeBlock
              code={install}
              wrap
              tabs={
                guide.language === "typescript" ? (
                  <FileLabel>npm</FileLabel>
                ) : (
                  <LineTabs value={installer} options={["pip", "uv"]} onChange={setInstaller} />
                )
              }
            />
          </Step>
        )}
        {guide.plugin && (
          <Step title="Enable the plugin">
            <p className="text-sm leading-6 text-muted-foreground">
              Install and enable the{" "}
              <a className="underline underline-offset-4" href={guide.plugin.url} target="_blank" rel="noreferrer">
                {guide.plugin.label}
              </a>
              . {guide.plugin.instruction}
            </p>
          </Step>
        )}
        <Step title="Configure environment">
          <p className="mb-3 text-sm leading-6 text-muted-foreground">
            {tracingKey ? (
              <>
                Tracing uses the key above. Set <code>LITELLM_API_KEY</code> to a key with model access.
              </>
            ) : (
              <>
                Set <code>LITELLM_API_KEY</code> to your LiteLLM key.
              </>
            )}
          </p>
          <CodeBlock
            code={tracingEnvSnippet(proxyUrl, tracingKey)}
            display={tracingEnvSnippet(proxyUrl, tracingKey && maskSecret(tracingKey))}
            tabs={<FileLabel>Shell</FileLabel>}
            wrap
          />
        </Step>
        <Step title="Run your agent">
          <p className="mb-3 text-sm leading-6 text-muted-foreground">
            {guide.existingModel ? "Keep your existing model settings. " : "Save and run this example. "}
            Replace <code>research_agent</code> with your agent’s name.
            {!guide.existingModel && " Use a model configured on this proxy."}
          </p>
          <CodeBlock code={quickstart} tabs={<FileLabel>{guide.fileName}</FileLabel>} wrap />
          {guide.note && <p className="mt-3 text-sm leading-6 text-muted-foreground">{guide.note}</p>}
          {!guide.existingModel && (
            <div className="mt-3">
              <CodeBlock
                code={guide.language === "typescript" ? "npx tsx agent.mts" : "python agent.py"}
                tabs={<FileLabel>Shell</FileLabel>}
              />
            </div>
          )}
          <a
            href={`${DOCS_URL}?framework=${guide.id}#send-your-first-trace`}
            className="mt-3 inline-flex items-center gap-1 text-sm underline underline-offset-4"
            target="_blank"
            rel="noreferrer"
          >
            View in docs <ArrowUpRight aria-hidden="true" className="size-3.5" />
          </a>
        </Step>
      </details>

      <TraceReceipt connected={connected} checked={checked} checking={checking} onCheck={onCheck} />
    </>
  );
}

type TracingSetupProps = {
  detail: string | null;
  accessToken: string;
  onOpenTrace: (trace: TraceSummary) => void;
  connected?: boolean;
  onCheck?: () => void;
  checking?: boolean;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
};

export function TracingSetupFields({
  detail,
  accessToken,
  onOpenTrace,
  connected = false,
  onCheck,
  checking = false,
  readOnly = false,
  canMintTracingKey = false,
}: TracingSetupProps) {
  const [checked, setChecked] = useState(false);
  const check = () => {
    setChecked(true);
    onCheck?.();
  };
  return detail === null ? (
    <ConnectAgent
      accessToken={accessToken}
      onOpenTrace={onOpenTrace}
      connected={connected}
      checked={checked}
      checking={checking}
      onCheck={check}
      readOnly={readOnly}
      canMintTracingKey={canMintTracingKey}
    />
  ) : (
    <EnableTracing checked={checked} checking={checking} onCheck={check} />
  );
}

export function TracingSetupCard(props: TracingSetupProps) {
  const enabled = props.detail === null;

  return (
    <div className="w-full max-w-3xl pb-8" data-testid="tracing-setup-card">
      {!props.connected && <LensPreviewButton />}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <h2 className="text-xl font-semibold tracking-tight">{setupTitle(enabled, props.connected ?? false)}</h2>
        <span role="status" className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
          {enabled ? (
            <ActiveDot />
          ) : (
            <span aria-hidden="true" className="size-1.5 rounded-full bg-muted-foreground/50" />
          )}
          {enabled ? "Tracing enabled" : "Tracing is not enabled"}
        </span>
        <a
          className="ml-auto inline-flex shrink-0 items-center gap-1 text-sm underline underline-offset-4"
          href={DOCS_URL}
          target="_blank"
          rel="noreferrer"
        >
          Docs <ArrowUpRight aria-hidden="true" className="size-3.5" />
        </a>
      </div>
      <p className="mt-2 text-sm leading-6 text-muted-foreground">
        {enabled
          ? "Send your agent’s runs to LiteLLM to see its inputs, outputs, and tool calls."
          : "Tracing needs ClickHouse and a small update to your LiteLLM proxy configuration."}
      </p>
      <TracingSetupFields {...props} />
    </div>
  );
}
