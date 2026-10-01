"use client";

import { ArrowUpRight, Check, Copy, Loader2 } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/cva.config";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
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
const DOCS_URL = "https://docs.litellm.ai/docs/proxy/lens";
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

const installPackages = (guide: Pick<FrameworkGuide, "packages">): string =>
  [OTEL_BASE_PACKAGES, guide.packages].filter(Boolean).join(" ");

/** The endpoint is the proxy base URL: OTLP exporters append /v1/traces themselves. */
export const tracingEnvSnippet = (proxyUrl: string): string =>
  [
    `export OTEL_EXPORTER_OTLP_ENDPOINT=${proxyUrl}`,
    "export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf",
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
    "   OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf",
    '   OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $LITELLM_API_KEY"',
    "   OTEL_SERVICE_NAME=<a short name for this agent>",
    "3. Start the app through OTEL auto-instrumentation: opentelemetry-instrument <existing start command>.",
    `4. Point every LLM client at LiteLLM: base_url=${proxyUrl}/v1, api key from LITELLM_API_KEY.`,
    "5. Give each agent and subagent a name so runs are easy to read.",
    "6. Run the agent once and confirm the run shows up in the LiteLLM UI under Lens > Traces.",
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

function setupTitle(enabled: boolean, connected: boolean) {
  if (!enabled) return "Enable tracing";
  return connected ? "Connect another agent" : "Connect your agent";
}

export function TracingSetupCard({
  detail,
  connected = false,
  onCheck,
  checking = false,
}: {
  detail: string | null;
  connected?: boolean;
  onCheck?: () => void;
  checking?: boolean;
}) {
  const proxyUrl = getProxyBaseUrl().replace(/\/$/, "");
  const [framework, setFramework] = useState(FRAMEWORKS[0].id);
  const [installer, setInstaller] = useState<Installer>("pip");
  const [copied, setCopied] = useState(false);
  const [checked, setChecked] = useState(false);
  const guide = FRAMEWORKS.find((f) => f.id === framework) ?? FRAMEWORKS[0];
  const packages = installPackages(guide);
  const install = installer === "pip" ? `pip install -U ${packages}` : `uv add ${packages}`;
  const enabled = detail === null;
  const copyPrompt = async () => {
    if (await copyToClipboard(codingAgentPrompt(proxyUrl, guide))) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), COPIED_RESET_MS);
    }
  };
  const check = () => {
    setChecked(true);
    onCheck?.();
  };

  return (
    <div className="mx-auto w-full max-w-2xl py-6 pb-12" data-testid="tracing-setup-card">
      <h2 className="text-xl font-semibold tracking-tight">{setupTitle(enabled, connected)}</h2>
      <p className="mt-2 text-sm leading-6 text-muted-foreground">
        {enabled
          ? "Send your agent’s runs to LiteLLM to see its inputs, outputs, and tool calls."
          : "Tracing needs ClickHouse and a small update to your LiteLLM proxy configuration."}
      </p>
      <p role="status" className="mt-4 flex items-center gap-2 text-sm">
        {enabled && <Check aria-hidden="true" className="size-4 text-emerald-600" />}
        {enabled ? "Tracing enabled" : "Tracing is not enabled"}
      </p>

      {!enabled ? (
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
          <Button className="mt-6" onClick={check} disabled={checking}>
            {checking && <Loader2 aria-hidden="true" className="size-4 animate-spin" />}
            {checking ? "Checking…" : "Check setup"}
          </Button>
        </>
      ) : (
        <>
          <div className="mt-6 space-y-2">
            <label id="tracing-framework" className="text-sm font-medium">
              Your agent framework
            </label>
            <Select
              value={framework}
              onValueChange={(value) => {
                if (value) {
                  setFramework(value);
                  setCopied(false);
                }
              }}
            >
              <SelectTrigger aria-labelledby="tracing-framework" className="w-full">
                <SelectValue>{guide.label}</SelectValue>
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
          <Step title="Let your coding agent connect it">
            <p className="mb-3 text-sm leading-6 text-muted-foreground">
              Paste the setup prompt into Claude Code or Codex in your agent’s project. It uses your existing LiteLLM
              key from the environment.
            </p>
            <Button variant="outline" onClick={() => void copyPrompt()}>
              {copied ? (
                <Check aria-hidden="true" className="size-4" />
              ) : (
                <Copy aria-hidden="true" className="size-4" />
              )}
              {copied ? "Prompt copied" : "Copy setup prompt"}
            </Button>
          </Step>
          <details className="mt-6 border-y py-4">
            <summary className="cursor-pointer text-sm font-medium">Set up manually</summary>
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
              <p className="mb-2 text-sm text-muted-foreground">
                Replace the example model with a model configured on your proxy.
              </p>
              <CodeBlock
                code={`import os\n\n${guide.quickstart.replace("{PROXY}", proxyUrl)}`}
                tabs={<FileLabel>my_agent.py</FileLabel>}
              />
              <div className="mt-3">
                <CodeBlock code={RUN_SNIPPET} tabs={<FileLabel>Shell</FileLabel>} />
              </div>
            </Step>
          </details>
          <TraceReceipt connected={connected} checked={checked} checking={checking} onCheck={check} />
          <details className="mt-6 text-sm">
            <summary className="cursor-pointer text-muted-foreground">See an example trace</summary>
            <p className="my-3 text-muted-foreground">Example only. These are not your agent’s runs.</p>
            <img
              src={previewImg.src}
              alt="Example agent trace with tool calls, inputs, and outputs"
              className="w-full rounded-md border"
            />
          </details>
        </>
      )}
    </div>
  );
}
