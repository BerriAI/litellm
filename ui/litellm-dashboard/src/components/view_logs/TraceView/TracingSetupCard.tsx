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

type Installer = "pip" | "uv";

interface FrameworkGuide {
  id: string;
  label: string;
  logo: string;
  packages: string;
  quickstart: string;
}

const FRAMEWORKS: readonly FrameworkGuide[] = [
  {
    id: "langgraph",
    label: "LangGraph / Deep Agents",
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

const HIGHLIGHTS = [
  ["Input and output", "What the agent was asked and what it answered, at the top of every run."],
  ["Every step, nested", "LLM calls, tool calls and subagents in one tree, with timing."],
  ["Failures pinpointed", "See whether the tool, the model or LiteLLM broke."],
  ["Hand off to Claude / Codex", "Copy one command and your coding agent debugs the run."],
] as const;

const installPackages = (guide: Pick<FrameworkGuide, "packages">): string =>
  [OTEL_BASE_PACKAGES, guide.packages].filter(Boolean).join(" ");

/** The endpoint is the proxy base URL: OTLP exporters append /v1/traces themselves. */
export const tracingEnvSnippet = (proxyUrl: string): string =>
  [
    `export OTEL_EXPORTER_OTLP_ENDPOINT=${proxyUrl}`,
    'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $LITELLM_API_KEY"',
    "export OTEL_SERVICE_NAME=my-agent",
  ].join("\n");

export const codingAgentPrompt = (proxyUrl: string, guide: Pick<FrameworkGuide, "label" | "packages">): string =>
  [
    `Send this ${guide.label} project's OpenTelemetry traces to LiteLLM.`,
    "",
    `1. Add these dependencies: ${installPackages(guide)}`,
    "2. Set these env vars wherever the project loads config (.env, settings, deployment manifests):",
    `   OTEL_EXPORTER_OTLP_ENDPOINT=${proxyUrl}`,
    '   OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $LITELLM_API_KEY"',
    "   OTEL_SERVICE_NAME=<a short name for this agent>",
    "3. Start the app through OTEL auto-instrumentation: opentelemetry-instrument <existing start command>.",
    `4. Point every LLM client at LiteLLM: base_url=${proxyUrl}/v1, api key from LITELLM_API_KEY.`,
    "5. Give each agent and subagent a name so runs are easy to read.",
    "6. Run the agent once and confirm the run shows up in the LiteLLM UI under Logs > Agent Traces.",
    "",
    "Never hardcode the key. Read it from LITELLM_API_KEY.",
  ].join("\n");

export const PROXY_CONFIG_SNIPPET = [
  "general_settings:",
  "  tracing:",
  "    store: clickhouse   # + CLICKHOUSE_URL",
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
}: {
  value: T;
  options: T[];
  onChange: (v: T) => void;
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
    <section className="mt-6">
      <h3 className="mb-2 text-[13px] font-medium text-foreground">{title}</h3>
      {children}
    </section>
  );
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

function FrameworkPicker({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  return (
    <div className="mt-4 flex flex-wrap gap-2" role="radiogroup" aria-label="Framework">
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

/**
 * Agent Traces onboarding: shown until the first trace arrives (and when tracing isn't enabled on the proxy).
 * `detail` is the proxy's 501 message when tracing is off; null means tracing is on and we're waiting.
 */
export function TracingSetupCard({ detail, connected = false }: { detail: string | null; connected?: boolean }) {
  const proxyUrl = getProxyBaseUrl().replace(/\/$/, "");
  const [framework, setFramework] = useState(FRAMEWORKS[0].id);
  const [installer, setInstaller] = useState<Installer>("pip");
  const guide = FRAMEWORKS.find((f) => f.id === framework) ?? FRAMEWORKS[0];
  const packages = installPackages(guide);
  const install = installer === "pip" ? `pip install -U ${packages}` : `uv add ${packages}`;

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
            Standard OTLP. Pick your framework, set three env vars, and run your agent as usual.
          </p>
          <FrameworkPicker value={framework} onChange={setFramework} />

          {detail !== null && (
            <Step title="Enable tracing on the proxy">
              <CodeBlock code={PROXY_CONFIG_SNIPPET} tabs={<FileLabel>config.yaml</FileLabel>} />
            </Step>
          )}

          <Step title="Let Claude Code or Codex set it up">
            <p className="mb-2 text-[13px] text-muted-foreground">
              Paste this into your coding agent from the project root, or follow the steps below by hand.
            </p>
            <CodeBlock code={codingAgentPrompt(proxyUrl, guide)} wrap tabs={<FileLabel>Prompt</FileLabel>} />
          </Step>

          <Step title="Install dependencies">
            <CodeBlock
              code={install}
              tabs={<LineTabs value={installer} options={["pip", "uv"]} onChange={setInstaller} />}
            />
          </Step>

          <Step title="Configure environment">
            <CodeBlock code={tracingEnvSnippet(proxyUrl)} tabs={<FileLabel>Shell</FileLabel>} />
          </Step>

          <Step title="Run your agent">
            <CodeBlock
              code={`import os\n\n${guide.quickstart.replace("{PROXY}", proxyUrl)}`}
              tabs={<FileLabel>my_agent.py</FileLabel>}
            />
            <div className="mt-3">
              <CodeBlock code={RUN_SNIPPET} tabs={<FileLabel>Shell</FileLabel>} />
            </div>
          </Step>
        </div>
      </div>
    </div>
  );
}
