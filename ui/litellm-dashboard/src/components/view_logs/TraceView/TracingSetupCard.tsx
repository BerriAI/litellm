"use client";

import { ArrowUpRight, Check, Copy, Loader2 } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import previewImg from "../../../../public/assets/agent-traces-preview.png";
import crewaiLogo from "../../../../public/assets/logos/crewai-color.svg";
import langchainLogo from "../../../../public/assets/logos/langchain.svg";
import langgraphLogo from "../../../../public/assets/logos/langgraph-color.svg";
import llamaindexLogo from "../../../../public/assets/logos/llamaindex-color.svg";
import openaiAgentsLogo from "../../../../public/assets/logos/openai-agents.svg";
import otelLogo from "../../../../public/assets/logos/opentelemetry.svg";
import pydanticAiLogo from "../../../../public/assets/logos/pydantic-ai-color.svg";
import { getProxyBaseUrl } from "../../networking";

const COPIED_RESET_MS = 1500;
const DOCS_URL = "https://docs.litellm.ai";
const OTEL_BASE_PACKAGES = "opentelemetry-distro opentelemetry-exporter-otlp-proto-http";
const RUN_SNIPPET = "opentelemetry-instrument python my_agent.py";
const TRACES_ROUTE = "/v1/traces";
// HTTP client instrumentors would otherwise emit a one-span "POST" run for every model call.
const DISABLED_INSTRUMENTATIONS = "httpx,requests,urllib3,urllib,aiohttp-client";
const LANGCHAIN_INSTRUMENTOR_PACKAGE = "openinference-instrumentation-langchain";

type Installer = "pip" | "uv";
type InstrumentMode = "Environment variables (no code)" | "In code";

interface FrameworkGuide {
  id: string;
  label: string;
  logo: string;
  packages: string;
  quickstart: string;
  /** The agent-building lines for the in-code walkthrough; only for frameworks whose instrumentor was verified. */
  inCodeAgent?: string;
}

const LLM_LINE = `llm = ChatOpenAI(model="claude-sonnet-4-5", base_url=f"{LITELLM_PROXY}/v1", api_key=LITELLM_API_KEY)`;

const FRAMEWORKS: readonly FrameworkGuide[] = [
  {
    id: "deepagents",
    label: "Deep Agents",
    logo: langgraphLogo.src,
    packages: `deepagents langchain-openai ${LANGCHAIN_INSTRUMENTOR_PACKAGE}`,
    quickstart: `from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="claude-sonnet-4-5", base_url="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"])
agent = create_deep_agent(model=llm, tools=[], name="my_agent")
agent.invoke({"messages": [{"role": "user", "content": "What is LiteLLM?"}]})`,
    inCodeAgent: `from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

${LLM_LINE}
agent = create_deep_agent(model=llm, tools=[], name="my_agent")
agent.invoke({"messages": [{"role": "user", "content": "What is LiteLLM?"}]})`,
  },
  {
    id: "langchain",
    label: "LangChain / LangGraph",
    logo: langchainLogo.src,
    packages: `langchain langgraph langchain-openai ${LANGCHAIN_INSTRUMENTOR_PACKAGE}`,
    quickstart: `from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="claude-sonnet-4-5", base_url="{PROXY}/v1", api_key=os.environ["LITELLM_API_KEY"])
agent = create_agent(model=llm, tools=[], name="my_agent")
agent.invoke({"messages": [{"role": "user", "content": "What is LiteLLM?"}]})`,
    inCodeAgent: `from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

${LLM_LINE}
agent = create_agent(model=llm, tools=[], name="my_agent")
agent.invoke({"messages": [{"role": "user", "content": "What is LiteLLM?"}]})`,
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

const HIGHLIGHTS = [
  ["Input and output", "What the agent was asked and what it answered, at the top of every run."],
  ["Every step, nested", "LLM calls, tool calls and subagents in one tree, with timing."],
  ["Failures pinpointed", "See whether the tool, the model or LiteLLM broke."],
  ["Hand off to Claude / Codex", "Copy one command and your coding agent debugs the run."],
] as const;

const INSTRUMENT_MODES: InstrumentMode[] = ["Environment variables (no code)", "In code"];

const installPackages = (guide: Pick<FrameworkGuide, "packages">): string =>
  [OTEL_BASE_PACKAGES, guide.packages].filter(Boolean).join(" ");

export interface EndpointRow {
  label: string;
  value: string;
}

/** What receives traces and how, for the guide and for any always-visible endpoint panel. */
export const otlpEndpointRows = (proxyUrl: string): EndpointRow[] => [
  { label: "OTLP endpoint", value: proxyUrl },
  { label: "Ingest route", value: `POST ${proxyUrl}${TRACES_ROUTE}` },
  { label: "Protocol", value: "OTLP/HTTP, protobuf (application/x-protobuf) or JSON" },
  {
    label: "Auth",
    value: "Authorization: Bearer <virtual key>. Traces are scoped to that key and its team",
  },
  { label: "List runs", value: `GET ${proxyUrl}${TRACES_ROUTE}?start_ms=…&end_ms=…` },
  { label: "One run", value: `GET ${proxyUrl}${TRACES_ROUTE}/{trace_id}` },
  { label: "Markdown for Claude / Codex", value: `GET ${proxyUrl}${TRACES_ROUTE}/{trace_id}?format=md` },
];

/** OTLP exporters append /v1/traces themselves, so the env endpoint is the proxy base URL. */
export const tracingEnvSnippet = (proxyUrl: string): string =>
  [
    `export OTEL_EXPORTER_OTLP_ENDPOINT=${proxyUrl}`,
    "export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf",
    'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $LITELLM_API_KEY"',
    "export OTEL_SERVICE_NAME=my-agent",
    `export OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=${DISABLED_INSTRUMENTATIONS}`,
  ].join("\n");

/** In code the exporter takes the full ingest URL, so /v1/traces is spelled out here. */
export const inCodeSnippet = (proxyUrl: string, agentLines: string): string =>
  [
    "import os",
    "",
    "from openinference.instrumentation.langchain import LangChainInstrumentor",
    "from opentelemetry import trace",
    "from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter",
    "from opentelemetry.sdk.resources import Resource",
    "from opentelemetry.sdk.trace import TracerProvider",
    "from opentelemetry.sdk.trace.export import BatchSpanProcessor",
    "",
    `LITELLM_PROXY = os.environ.get("LITELLM_PROXY", "${proxyUrl}")`,
    'LITELLM_API_KEY = os.environ["LITELLM_API_KEY"]',
    "",
    'provider = TracerProvider(resource=Resource.create({"service.name": "my-agent"}))',
    "provider.add_span_processor(",
    "    BatchSpanProcessor(",
    "        OTLPSpanExporter(",
    `            endpoint=f"{LITELLM_PROXY}${TRACES_ROUTE}",`,
    '            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},',
    "        )",
    "    )",
    ")",
    "trace.set_tracer_provider(provider)",
    "LangChainInstrumentor().instrument(tracer_provider=provider)",
    "",
    agentLines,
    "",
    "provider.shutdown()  # flush spans before the process exits",
  ].join("\n");

export const modelClientSnippet = (proxyUrl: string): string =>
  [
    "from langchain_openai import ChatOpenAI",
    "",
    `llm = ChatOpenAI(model="claude-sonnet-4-5", base_url="${proxyUrl}/v1", api_key=os.environ["LITELLM_API_KEY"])`,
  ].join("\n");

export const verifySnippet = (proxyUrl: string): string =>
  [
    "NOW=$(($(date +%s) * 1000))",
    `curl "${proxyUrl}${TRACES_ROUTE}?start_ms=$((NOW - 3600000))&end_ms=$NOW" \\`,
    '  -H "Authorization: Bearer $LITELLM_API_KEY"',
    "",
    '# -> {"data": [{"trace_id": "...", "name": "my_agent", "span_count": ...}], "next_cursor": null}',
  ].join("\n");

export const codingAgentPrompt = (proxyUrl: string, guide: Pick<FrameworkGuide, "label" | "packages">): string =>
  [
    `Send this ${guide.label} project's OpenTelemetry traces to LiteLLM, then prove it works.`,
    "",
    `1. Add these dependencies: ${installPackages(guide)}`,
    "2. Set these env vars wherever the project loads config (.env, settings, deployment manifests):",
    `   OTEL_EXPORTER_OTLP_ENDPOINT=${proxyUrl}   (base URL only; the exporter appends ${TRACES_ROUTE})`,
    "   OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf",
    '   OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $LITELLM_API_KEY"',
    "   OTEL_SERVICE_NAME=<a short name for this agent>",
    `   OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=${DISABLED_INSTRUMENTATIONS}`,
    "3. Start the app through OTEL auto-instrumentation: opentelemetry-instrument <existing start command>.",
    `   If the project can't be started that way, set up a TracerProvider with OTLPSpanExporter(endpoint="${proxyUrl}${TRACES_ROUTE}")`,
    "   and call the framework's OpenInference instrumentor at startup instead.",
    `4. Point every LLM client at LiteLLM: base_url=${proxyUrl}/v1, api key from LITELLM_API_KEY, so each step has real cost.`,
    "5. Pass name=... to every agent and subagent so runs are easy to read.",
    "6. Flush spans before short-lived scripts exit (provider.shutdown() or force_flush()).",
    `7. Verify: run the agent once, then GET ${proxyUrl}${TRACES_ROUTE}?start_ms=<now-1h in ms>&end_ms=<now in ms>`,
    '   with header "Authorization: Bearer $LITELLM_API_KEY" and confirm the run appears in "data".',
    "",
    "Never hardcode the key. Read it from LITELLM_API_KEY.",
  ].join("\n");

export const PROXY_CONFIG_SNIPPET = [
  "general_settings:",
  "  tracing:",
  "    store: clickhouse",
  "",
  "# env: CLICKHOUSE_URL (writer) and CLICKHOUSE_READER_URL (read-only user)",
].join("\n");

function CodeBlock({ code, tabs, wrap = false }: { code: string; tabs?: React.ReactNode; wrap?: boolean }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await copyToClipboard(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), COPIED_RESET_MS);
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
        <code>{code}</code>
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
  label,
}: {
  value: T;
  options: T[];
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex h-9 items-end gap-3">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          role="tab"
          aria-selected={value === option}
          onClick={() => onChange(option)}
          className={cn(
            "-mb-px h-9 border-b-2 text-[12.5px]",
            value === option
              ? "border-foreground text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground",
          )}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

function Step({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-6" aria-label={title}>
      <h3 className="mb-2 text-[13px] font-medium text-foreground">{title}</h3>
      {children}
    </section>
  );
}

function Hint({ children }: { children: React.ReactNode }) {
  return <p className="mb-2 text-[13px] text-muted-foreground">{children}</p>;
}

function SetupStatus({ detail, connected }: { detail: string | null; connected: boolean }) {
  const pill = "inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-[12px] text-foreground";
  const hint = "text-[13px] text-muted-foreground";
  if (connected) {
    return (
      <>
        <span className={pill}>
          <Check className="size-3" /> Receiving traces
        </span>
        <span className={hint}>Add another agent: point its OpenTelemetry exporter at LiteLLM.</span>
      </>
    );
  }
  if (detail === null) {
    return (
      <>
        <span className={pill}>
          <Loader2 className="size-3 animate-spin" /> Waiting for traces…
        </span>
        <span className={hint}>No traces detected yet. Follow our guide to start tracing your application.</span>
      </>
    );
  }
  return (
    <>
      <span className={pill}>Tracing is not enabled</span>
      <span className={hint}>Turn on tracing in the proxy config, then point your agent at LiteLLM.</span>
    </>
  );
}

function WhatYoullSee() {
  return (
    <section className="mt-6">
      <h2 className="text-[15px] font-medium text-foreground">What you&apos;ll see</h2>
      <div className="mt-3 overflow-hidden rounded-lg border border-border bg-background shadow-sm">
        <img src={previewImg.src} alt="Preview of an agent trace in LiteLLM" className="block w-full" />
      </div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {HIGHLIGHTS.map(([title, body]) => (
          <div key={title}>
            <div className="text-[13px] font-medium text-foreground">{title}</div>
            <div className="mt-0.5 text-[13px] text-muted-foreground">{body}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function EndpointTable({ proxyUrl }: { proxyUrl: string }) {
  return (
    <dl className="overflow-hidden rounded-md border border-border" data-testid="otlp-endpoint">
      {otlpEndpointRows(proxyUrl).map((row) => (
        <div key={row.label} className="grid grid-cols-[180px_1fr] border-b border-border last:border-0">
          <dt className="bg-muted/40 px-3 py-2 text-[12.5px] text-muted-foreground">{row.label}</dt>
          <dd className="min-w-0 px-3 py-2 font-mono text-[12.5px] break-all text-foreground">{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function FrameworkPicker({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  return (
    <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Framework">
      {FRAMEWORKS.map((f) => (
        <button
          key={f.id}
          type="button"
          role="radio"
          aria-checked={value === f.id}
          onClick={() => onChange(f.id)}
          className={cn(
            "inline-flex items-center gap-2 rounded-md border px-3 py-2 text-[13px]",
            value === f.id
              ? "border-foreground bg-accent text-foreground"
              : "border-border text-muted-foreground hover:text-foreground",
          )}
        >
          <img src={f.logo} alt="" className="size-4" />
          {f.label}
        </button>
      ))}
    </div>
  );
}

function EnvInstrumentation({
  guide,
  proxyUrl,
  installer,
  onInstallerChange,
}: {
  guide: FrameworkGuide;
  proxyUrl: string;
  installer: Installer;
  onInstallerChange: (installer: Installer) => void;
}) {
  const packages = installPackages(guide);
  const install = installer === "pip" ? `pip install -U ${packages}` : `uv add ${packages}`;
  return (
    <div className="grid gap-3">
      <CodeBlock
        code={install}
        tabs={<LineTabs value={installer} options={["pip", "uv"]} onChange={onInstallerChange} />}
      />
      <Hint>
        The endpoint is the proxy base URL; the exporter adds {TRACES_ROUTE}. The last line stops HTTP client
        instrumentation from logging every model call as its own run.
      </Hint>
      <CodeBlock code={tracingEnvSnippet(proxyUrl)} tabs={<FileLabel>Shell</FileLabel>} />
      <CodeBlock
        code={`import os\n\n${guide.quickstart.replace("{PROXY}", proxyUrl)}`}
        tabs={<FileLabel>my_agent.py</FileLabel>}
      />
      <CodeBlock code={RUN_SNIPPET} tabs={<FileLabel>Shell</FileLabel>} />
    </div>
  );
}

function CodeInstrumentation({ guide, proxyUrl }: { guide: FrameworkGuide; proxyUrl: string }) {
  if (!guide.inCodeAgent) return null;
  return (
    <div className="grid gap-3">
      <CodeBlock code={`pip install -U ${installPackages(guide)}`} tabs={<FileLabel>Shell</FileLabel>} />
      <Hint>
        {guide.label} runs on LangChain, so LangChainInstrumentor traces every agent, LLM call and tool call. In code
        the exporter takes the full ingest URL, ending in {TRACES_ROUTE}. Run it with plain python.
      </Hint>
      <CodeBlock code={inCodeSnippet(proxyUrl, guide.inCodeAgent)} tabs={<FileLabel>my_agent.py</FileLabel>} />
    </div>
  );
}

/**
 * Agent Traces onboarding: shown until the first trace arrives (and when tracing isn't enabled on the proxy).
 * `detail` is the proxy's 501 message when tracing is off; null means tracing is on and we're waiting.
 */
export function TracingSetupCard({ detail, connected = false }: { detail: string | null; connected?: boolean }) {
  const proxyUrl = getProxyBaseUrl().replace(/\/$/, "");
  const [framework, setFramework] = useState(FRAMEWORKS[0].id);
  const [installer, setInstaller] = useState<Installer>("pip");
  const [mode, setMode] = useState<InstrumentMode>(INSTRUMENT_MODES[0]);
  const guide = FRAMEWORKS.find((f) => f.id === framework) ?? FRAMEWORKS[0];
  const modes = guide.inCodeAgent ? INSTRUMENT_MODES : INSTRUMENT_MODES.slice(0, 1);
  const activeMode = modes.includes(mode) ? mode : modes[0];

  return (
    <div className="pb-12" data-testid="tracing-setup-card">
      <div className="w-full rounded-lg border border-border bg-card p-6">
        <div className="flex flex-wrap items-center gap-3">
          <SetupStatus detail={detail} connected={connected} />
          <a
            href={DOCS_URL}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex items-center gap-1 text-[13px] text-foreground hover:underline"
          >
            View docs <ArrowUpRight className="size-3.5" />
          </a>
        </div>

        <WhatYoullSee />

        <div className="mt-10 max-w-[860px]">
          <h2 className="text-[15px] font-medium text-foreground">
            Send your agent&apos;s OpenTelemetry traces to LiteLLM
          </h2>
          <p className="mt-1 text-[13px] text-muted-foreground">
            Standard OTLP. No vendor SDK: point any OpenTelemetry exporter at your LiteLLM proxy.
          </p>

          {detail !== null && (
            <Step title="Enable tracing on the proxy">
              <CodeBlock code={PROXY_CONFIG_SNIPPET} tabs={<FileLabel>config.yaml</FileLabel>} />
            </Step>
          )}

          <Step title="OpenTelemetry endpoint">
            <EndpointTable proxyUrl={proxyUrl} />
          </Step>

          <Step title="Pick your framework">
            <FrameworkPicker value={framework} onChange={setFramework} />
          </Step>

          <Step title="Instrument your agent">
            <div className="mb-3 border-b border-border">
              <LineTabs value={activeMode} options={modes} onChange={setMode} label="Instrumentation method" />
            </div>
            {activeMode === "In code" ? (
              <CodeInstrumentation guide={guide} proxyUrl={proxyUrl} />
            ) : (
              <EnvInstrumentation
                guide={guide}
                proxyUrl={proxyUrl}
                installer={installer}
                onInstallerChange={setInstaller}
              />
            )}
          </Step>

          <Step title="Point model calls at LiteLLM">
            <Hint>
              Send the agent&apos;s model calls through the proxy too, so every LLM step carries its real cost.
            </Hint>
            <CodeBlock code={modelClientSnippet(proxyUrl)} tabs={<FileLabel>Python</FileLabel>} />
          </Step>

          <Step title="Let Claude Code or Codex set it up">
            <Hint>Paste this into your coding agent from the project root. It wires everything up and checks it.</Hint>
            <CodeBlock code={codingAgentPrompt(proxyUrl, guide)} wrap tabs={<FileLabel>Prompt</FileLabel>} />
          </Step>

          <Step title="Verify">
            <Hint>Run the agent once. The run appears in this tab within a few seconds, and the API returns it:</Hint>
            <CodeBlock code={verifySnippet(proxyUrl)} tabs={<FileLabel>Shell</FileLabel>} />
          </Step>
        </div>
      </div>
    </div>
  );
}
