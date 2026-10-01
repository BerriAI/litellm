"use client";

import { ArrowRight, ArrowUpRight, Check, Copy, KeyRound, Loader2, Send } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/cva.config";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { copyToClipboard } from "@/utils/dataUtils";

import anthropicLogo from "../../../../public/assets/logos/anthropic.svg";
import crewaiLogo from "../../../../public/assets/logos/crewai-color.svg";
import langchainLogo from "../../../../public/assets/logos/langchain.svg";
import langgraphLogo from "../../../../public/assets/logos/langgraph-color.svg";
import llamaindexLogo from "../../../../public/assets/logos/llamaindex-color.svg";
import openaiAgentsLogo from "../../../../public/assets/logos/openai-agents.svg";
import openaiLogo from "../../../../public/assets/logos/openai_small.svg";
import otelLogo from "../../../../public/assets/logos/opentelemetry.svg";
import pydanticAiLogo from "../../../../public/assets/logos/pydantic-ai-color.svg";
import vercelLogo from "../../../../public/assets/logos/vercel.svg";
import { agentTraceCall, apiClient, getProxyBaseUrl, sendOtlpTraceCall } from "../../networking";
import { ActiveDot } from "./ActiveDot";
import { sampleTraceExport } from "./sampleTrace";
import { TracePreview } from "./TracePreview";
import type { TraceSummary } from "./traceTypes";

const COPIED_RESET_MS = 1500;
const DOCS_URL = "https://docs.litellm.ai/docs/proxy/lens";
const OTEL_BASE_PACKAGES = "opentelemetry-distro opentelemetry-exporter-otlp-proto-http";
const PY_RUN_SNIPPET = "opentelemetry-instrument python my_agent.py";
const TS_RUN_SNIPPET = "npx tsx my_agent.ts";
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
  pip: (packages) => `pip install -U ${packages}`,
  uv: (packages) => `uv add ${packages}`,
};

interface FrameworkGuide {
  id: string;
  label: string;
  logo: string;
  packages: string;
  quickstart: string;
  typescript?: boolean;
}

const FRAMEWORKS: readonly FrameworkGuide[] = [
  {
    id: "deep-agents",
    label: "Deep Agents",
    logo: langgraphLogo.src,
    packages: "deepagents langchain-openai openinference-instrumentation-langchain",
    quickstart: `from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="claude-sonnet-4-5", base_url="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"])
agent = create_deep_agent(model=llm, tools=[], system_prompt="You are a careful researcher.")
agent.invoke({"messages": [{"role": "user", "content": "What is LiteLLM?"}]})`,
  },
  {
    id: "vercel-ai-sdk",
    label: "Vercel AI SDK",
    logo: vercelLogo.src,
    typescript: true,
    packages: "ai @ai-sdk/openai-compatible @vercel/otel @opentelemetry/api",
    quickstart: `import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { registerOTel } from "@vercel/otel";
import { generateText } from "ai";

registerOTel({ serviceName: process.env.OTEL_SERVICE_NAME ?? "my-agent" });

const litellm = createOpenAICompatible({ name: "litellm", baseURL: "{PROXY}/v1", apiKey: process.env.LITELLM_API_KEY });

async function main() {
  const { text } = await generateText({
    model: litellm("claude-sonnet-4-5"),
    prompt: "What is LiteLLM?",
    experimental_telemetry: { isEnabled: true, functionId: "my_agent" },
  });
  console.log(text);
}

main();`,
  },
  {
    id: "langgraph",
    label: "LangGraph",
    logo: langgraphLogo.src,
    packages: "langgraph langchain-openai openinference-instrumentation-langchain",
    quickstart: `from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="claude-sonnet-4-5", base_url="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"])
agent = create_agent(model=llm, tools=[], name="my_agent")
agent.invoke({"messages": [{"role": "user", "content": "What is LiteLLM?"}]})`,
  },
  {
    id: "langchain",
    label: "LangChain",
    logo: langchainLogo.src,
    packages: "langchain langchain-openai openinference-instrumentation-langchain",
    quickstart: `from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="claude-sonnet-4-5", base_url="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"])
llm.invoke("What is LiteLLM?")`,
  },
  {
    id: "openai-agents",
    label: "OpenAI Agents SDK",
    logo: openaiAgentsLogo.src,
    packages: "openai-agents openinference-instrumentation-openai-agents",
    quickstart: `from agents import Agent, OpenAIChatCompletionsModel, Runner
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"])
agent = Agent(name="my_agent", model=OpenAIChatCompletionsModel(model="claude-sonnet-4-5", openai_client=client))
print(Runner.run_sync(agent, "What is LiteLLM?").final_output)`,
  },
  {
    id: "crewai",
    label: "CrewAI",
    logo: crewaiLogo.src,
    packages: "crewai openinference-instrumentation-crewai",
    quickstart: `from crewai import LLM, Agent, Crew, Task

llm = LLM(model="openai/claude-sonnet-4-5", base_url="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"])
agent = Agent(role="Researcher", goal="Answer questions", backstory="", llm=llm)
task = Task(description="What is LiteLLM?", expected_output="A short answer", agent=agent)
Crew(agents=[agent], tasks=[task]).kickoff()`,
  },
  {
    id: "pydantic-ai",
    label: "Pydantic AI",
    logo: pydanticAiLogo.src,
    packages: "pydantic-ai openinference-instrumentation-pydantic-ai",
    quickstart: `from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(base_url="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"])
agent = Agent(OpenAIModel("claude-sonnet-4-5", provider=provider), name="my_agent", instrument=True)
print(agent.run_sync("What is LiteLLM?").output)`,
  },
  {
    id: "llamaindex",
    label: "LlamaIndex",
    logo: llamaindexLogo.src,
    packages: "llama-index llama-index-llms-openai-like openinference-instrumentation-llama-index",
    quickstart: `from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(model="claude-sonnet-4-5", api_base="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"], is_chat_model=True)
print(llm.complete("What is LiteLLM?"))`,
  },
  {
    id: "otel",
    label: "OpenTelemetry",
    logo: otelLogo.src,
    packages: "",
    quickstart: `# Any OTEL SDK works. Use the gen_ai.* semantic conventions:
# gen_ai.operation.name, gen_ai.agent.name, gen_ai.response.id, gen_ai.usage.*
from opentelemetry import trace

tracer = trace.get_tracer("my-agent")
attrs = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "my_agent"}
with tracer.start_as_current_span("my_agent", attributes=attrs):
    ...`,
  },
];

const installPackages = (guide: Pick<FrameworkGuide, "packages" | "typescript">): string =>
  guide.typescript ? guide.packages : [OTEL_BASE_PACKAGES, guide.packages].filter(Boolean).join(" ");

/** The endpoint is the proxy base URL: OTLP exporters append /v1/traces themselves. */
export const tracingEnvSnippet = (proxyUrl: string, tracingKey: string | null = null): string =>
  [
    ...(tracingKey ? [`export LITELLM_TRACING_KEY=${tracingKey}`] : []),
    `export OTEL_EXPORTER_OTLP_ENDPOINT=${proxyUrl}`,
    "export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf",
    `export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $${tracingKey ? "LITELLM_TRACING_KEY" : "LITELLM_API_KEY"}"`,
    "export OTEL_SERVICE_NAME=my-agent",
  ].join("\n");

export const codingAgentPrompt = (
  proxyUrl: string,
  guide: Pick<FrameworkGuide, "label" | "packages" | "typescript">,
): string =>
  [
    `Send this ${guide.label} project's OpenTelemetry traces to LiteLLM.`,
    "",
    `1. Add these dependencies: ${installPackages(guide)}`,
    "2. Set these env vars wherever the project loads config (.env, settings, deployment manifests):",
    `   OTEL_EXPORTER_OTLP_ENDPOINT=${proxyUrl}`,
    "   OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf",
    '   OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $LITELLM_API_KEY"',
    "   OTEL_SERVICE_NAME=<a short name for this agent>",
    guide.typescript
      ? "3. Call registerOTel() from @vercel/otel at startup and pass experimental_telemetry: { isEnabled: true } to every AI SDK call."
      : "3. Start the app through OTEL auto-instrumentation: opentelemetry-instrument <existing start command>.",
    `4. Point every LLM client at LiteLLM: base_url=${proxyUrl}/v1, api key from LITELLM_API_KEY.`,
    "5. Give each agent and subagent a name so runs are easy to read.",
    "6. Run the agent once and confirm the run shows up in the LiteLLM UI under Lens > Traces.",
    "",
    "Never hardcode the key. Read it from LITELLM_API_KEY.",
  ].join("\n");

const shellQuote = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`;

export const codingAgentCommand = (agent: CodingAgent, prompt: string): string =>
  `${agent === "Claude Code" ? "claude" : "codex"} ${shellQuote(prompt)}`;

export const maskSecret = (secret: string): string =>
  secret.length > 10 ? `${secret.slice(0, 5)}${"•".repeat(16)}${secret.slice(-4)}` : "•".repeat(secret.length);

export const otlpEndpoints = (proxyUrl: string): readonly (readonly [string, string, boolean])[] => [
  ["Traces endpoint (POST)", `${proxyUrl}/v1/traces`, true],
  ["OTEL_EXPORTER_OTLP_ENDPOINT", proxyUrl, true],
  ["Auth header", "Authorization: Bearer <LiteLLM virtual key>", true],
  ["Protocol", "OTLP/HTTP, protobuf or JSON (gRPC not supported)", false],
];

export const PROXY_CONFIG_SNIPPET = [
  "general_settings:",
  "  tracing:",
  "    store: clickhouse",
  "",
  "# env: CLICKHOUSE_URL (writer) and CLICKHOUSE_READER_URL (read-only user)",
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
  const copy = async () => {
    if (await copyToClipboard(code)) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), COPIED_RESET_MS);
    }
  };
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
          "px-4 py-3 font-mono text-[12.5px] leading-6 text-foreground",
          wrap ? "whitespace-pre-wrap" : "overflow-x-auto",
        )}
      >
        <code>{display}</code>
      </pre>
    </div>
  );
}

function FileLabel({ children }: { children: React.ReactNode }) {
  return <span className="text-[12.5px] text-foreground">{children}</span>;
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
            "-mb-px inline-flex h-9 items-center gap-1.5 border-b-2 text-[12.5px]",
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
      <h3 className="mb-2 text-[13px] font-medium text-foreground">{title}</h3>
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
      <Button onClick={() => void send()} disabled={busy}>
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
    <Step title="Run your agent once">
      <p className="text-sm leading-6 text-muted-foreground">
        Keep this page open. Your runs will appear here when LiteLLM receives them.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button variant="outline" onClick={onCheck} disabled={checking}>
          {checking && <Loader2 aria-hidden="true" className="size-4 animate-spin" />}
          {checking ? "Checking…" : "Check for traces"}
        </Button>
        {!connected && (
          <p role="status" className="text-sm text-muted-foreground">
            Waiting for your first trace
          </p>
        )}
      </div>
      {missingAfterCheck && (
        <p className="mt-3 text-sm leading-6 text-muted-foreground">
          No traces received yet. Check the exporter URL and LiteLLM key in your agent’s environment, then check its
          logs for export errors.
        </p>
      )}
    </Step>
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
  const copy = async () => {
    if (await copyToClipboard(value)) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), COPIED_RESET_MS);
    }
  };
  return (
    <button
      type="button"
      onClick={() => void copy()}
      aria-label={`Copy ${value}`}
      className="group inline-flex min-w-0 items-center gap-2 text-left font-mono text-[12.5px] text-foreground"
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

function Endpoints({ proxyUrl }: { proxyUrl: string }) {
  return (
    <section className="mt-6" aria-labelledby="otel-endpoints">
      <h3 id="otel-endpoints" className="inline-flex items-center gap-2 text-sm font-medium">
        <img src={otelLogo.src} alt="" className="size-5" />
        OpenTelemetry (OTEL) endpoints
      </h3>
      <p className="mt-1 text-sm text-muted-foreground">
        Point any OpenTelemetry exporter here. The framework guides below set these for you.
      </p>
      <dl className="mt-3 grid gap-x-8 gap-y-3 rounded-md border border-l-2 border-border border-l-trace-brand bg-muted/30 px-4 py-3 md:grid-cols-2">
        {otlpEndpoints(proxyUrl).map(([label, value, copyable]) => (
          <div key={label} className="min-w-0">
            <dt className="text-[12px] text-muted-foreground">{label}</dt>
            <dd className="mt-0.5">
              {copyable ? <EndpointValue value={value} /> : <span className="text-[12.5px]">{value}</span>}
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
      <Step title="Enable tracing on the proxy">
        <p className="mb-3 text-sm leading-6 text-muted-foreground">
          Set your ClickHouse writer and read-only reader URLs, add this to config.yaml, then restart the proxy. Ask
          your proxy administrator if you don’t manage this deployment.
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
      <Button className="mt-6" onClick={onCheck} disabled={checking}>
        {checking && <Loader2 aria-hidden="true" className="size-4 animate-spin" />}
        {checking ? "Checking…" : "Check setup"}
      </Button>
    </>
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
  const [codingAgent, setCodingAgent] = useState<CodingAgent>("Claude Code");
  const [tracingKey, setTracingKey] = useState<string | null>(null);
  const guide = FRAMEWORKS.find((f) => f.id === framework) ?? FRAMEWORKS[0];
  const packages = installPackages(guide);
  const install = guide.typescript ? `npm install ${packages}` : PY_INSTALL[installer](packages);
  const quickstart = guide.quickstart.replace("{PROXY}", proxyUrl);
  return (
    <>
      {!readOnly && (
        <Step title="See it work in one click">
          <p className="mb-3 text-sm leading-6 text-muted-foreground">
            Send a small sample run (an agent, an LLM call and a tool call) to confirm tracing works end to end.
          </p>
          <SendTestTrace accessToken={accessToken} onOpenTrace={onOpenTrace} />
        </Step>
      )}

      <Endpoints proxyUrl={proxyUrl} />

      <div className="mt-6 space-y-2">
        <label id="tracing-framework" className="text-sm font-medium">
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
              <img src={guide.logo} alt="" className="size-4" />
              {guide.label}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {FRAMEWORKS.map((f) => (
              <SelectItem key={f.id} value={f.id}>
                <img src={f.logo} alt="" className="size-4" />
                {f.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Step title="Get a LiteLLM key">
        {canMintTracingKey && !readOnly ? (
          <TracingKey accessToken={accessToken} tracingKey={tracingKey} onCreated={setTracingKey} />
        ) : (
          <p className="text-sm text-muted-foreground">
            Use any LiteLLM virtual key you already have, or ask a proxy admin for one.
          </p>
        )}
      </Step>

      <Step
        title={
          <span className="inline-flex items-center gap-1.5">
            Let <img src={anthropicLogo.src} alt="" className="size-3.5" /> Claude Code or
            <img src={openaiLogo.src} alt="" className="size-3.5" /> Codex connect it
          </span>
        }
      >
        <p className="mb-3 text-sm leading-6 text-muted-foreground">
          Run this in your agent’s project. It starts your coding agent with the setup task and reads the key from
          LITELLM_API_KEY.
        </p>
        <CodeBlock
          code={codingAgentCommand(codingAgent, codingAgentPrompt(proxyUrl, guide))}
          wrap
          tabs={
            <LineTabs
              value={codingAgent}
              options={["Claude Code", "Codex"]}
              onChange={setCodingAgent}
              logos={CODING_AGENT_LOGOS}
            />
          }
        />
      </Step>

      <details className="mt-6 border-y py-4">
        <summary className="cursor-pointer text-sm font-medium">Set up manually</summary>
        <Step title="Install dependencies">
          <CodeBlock
            code={install}
            tabs={
              guide.typescript ? (
                <FileLabel>npm</FileLabel>
              ) : (
                <LineTabs value={installer} options={["pip", "uv"]} onChange={setInstaller} />
              )
            }
          />
        </Step>
        <Step title="Configure environment">
          <CodeBlock
            code={tracingEnvSnippet(proxyUrl, tracingKey)}
            display={tracingEnvSnippet(proxyUrl, tracingKey && maskSecret(tracingKey))}
            tabs={<FileLabel>Shell</FileLabel>}
          />
        </Step>
        <Step title="Run your agent">
          <p className="mb-2 text-sm text-muted-foreground">
            Replace the example model with a model configured on your proxy.
          </p>
          <CodeBlock
            code={guide.typescript ? quickstart : `import os\n\n${quickstart}`}
            tabs={<FileLabel>{guide.typescript ? "my_agent.ts" : "my_agent.py"}</FileLabel>}
          />
          <div className="mt-3">
            <CodeBlock code={guide.typescript ? TS_RUN_SNIPPET : PY_RUN_SNIPPET} tabs={<FileLabel>Shell</FileLabel>} />
          </div>
        </Step>
      </details>

      <TraceReceipt connected={connected} checked={checked} checking={checking} onCheck={onCheck} />

      {!connected && (
        <section className="mt-6">
          <h3 className="mb-2 text-sm font-medium">
            Example run <span className="font-normal text-muted-foreground">(sample data, not your runs)</span>
          </h3>
          <TracePreview />
        </section>
      )}
    </>
  );
}

export function TracingSetupCard({
  detail,
  accessToken,
  onOpenTrace,
  connected = false,
  onCheck,
  checking = false,
  readOnly = false,
  canMintTracingKey = false,
}: {
  detail: string | null;
  accessToken: string;
  onOpenTrace: (trace: TraceSummary) => void;
  connected?: boolean;
  onCheck?: () => void;
  checking?: boolean;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
}) {
  const [checked, setChecked] = useState(false);
  const enabled = detail === null;
  const check = () => {
    setChecked(true);
    onCheck?.();
  };

  return (
    <div className="w-full py-6 pb-12" data-testid="tracing-setup-card">
      <div className="flex items-start gap-4">
        <h2 className="text-xl font-semibold tracking-tight">{setupTitle(enabled, connected)}</h2>
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
      <p role="status" className="mt-4 flex items-center gap-2 text-sm">
        {enabled && <ActiveDot />}
        {enabled ? "Tracing enabled" : "Tracing is not enabled"}
      </p>
      {enabled ? (
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
      )}
    </div>
  );
}
