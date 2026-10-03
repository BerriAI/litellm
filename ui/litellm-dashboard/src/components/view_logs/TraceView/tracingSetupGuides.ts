import langgraphLogo from "../../../../public/assets/logos/langgraph-color.svg";
import langchainLogo from "../../../../public/assets/logos/langchain.svg";
import openaiAgentsLogo from "../../../../public/assets/logos/openai-agents.svg";
import anthropicLogo from "../../../../public/assets/logos/anthropic.svg";
import crewaiLogo from "../../../../public/assets/logos/crewai-color.svg";
import pydanticAiLogo from "../../../../public/assets/logos/pydantic-ai-color.svg";
import llamaindexLogo from "../../../../public/assets/logos/llamaindex-color.svg";
import vercelLogo from "../../../../public/assets/logos/vercel.svg";
import otelLogo from "../../../../public/assets/logos/opentelemetry.svg";

import adkLogo from "../../../../public/assets/logos/google-adk.png";
import strandsLogo from "../../../../public/assets/logos/strands.svg";
import hermesLogo from "../../../../public/assets/logos/hermes.png";
import openclawLogo from "../../../../public/assets/logos/openclaw.png";

export interface FrameworkGuide {
  id: string;
  label: string;
  logo: string;
  install?: string;
  language: "python" | "typescript" | "json" | "yaml";
  quickstart: string;
  fileName: string;
  existingModel?: boolean;
  note?: string;
  plugin?: { label: string; url: string; instruction: string };
}

export const FRAMEWORKS: readonly FrameworkGuide[] = [
  {
    id: "deepagents",
    label: "DeepAgents",
    logo: langgraphLogo.src,
    install:
      "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http deepagents openinference-instrumentation-langchain langchain-openai",
    language: "python",
    quickstart: `import os

AGENT_NAME = "research_agent"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from deepagents import create_deep_agent

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model={MODEL},
    base_url={BASE_URL},
    api_key=os.environ["LITELLM_API_KEY"],
)

agent = create_deep_agent(name=AGENT_NAME, model=model, tools=[])
result = agent.invoke({"messages": [{"role": "user", "content": "What is an agent trace?"}]})
print(result["messages"][-1].content)`,
    fileName: "agent.py",
  },
  {
    id: "langgraph",
    label: "LangGraph",
    logo: langgraphLogo.src,
    install:
      "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http langgraph openinference-instrumentation-langchain langchain-openai",
    language: "python",
    quickstart: `import os

AGENT_NAME = "research_agent"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model={MODEL},
    base_url={BASE_URL},
    api_key=os.environ["LITELLM_API_KEY"],
)
graph = StateGraph(MessagesState)
graph.add_node("answer", lambda state: {"messages": [model.invoke(state["messages"])]})
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
agent = graph.compile(name=AGENT_NAME)
result = agent.invoke({"messages": [{"role": "user", "content": "What is an agent trace?"}]})
print(result["messages"][-1].content)`,
    fileName: "agent.py",
  },
  {
    id: "langchain",
    label: "LangChain",
    logo: langchainLogo.src,
    install:
      "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http langchain openinference-instrumentation-langchain langchain-openai",
    language: "python",
    quickstart: `import os

AGENT_NAME = "research_agent"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from langchain.agents import create_agent

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model={MODEL},
    base_url={BASE_URL},
    api_key=os.environ["LITELLM_API_KEY"],
)

agent = create_agent(name=AGENT_NAME, model=model, tools=[])
result = agent.invoke({"messages": [{"role": "user", "content": "What is an agent trace?"}]})
print(result["messages"][-1].content)`,
    fileName: "agent.py",
  },
  {
    id: "openai-agents",
    label: "OpenAI Agents",
    logo: openaiAgentsLogo.src,
    install:
      "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http openai-agents openinference-instrumentation-openai-agents",
    language: "python",
    quickstart: `import os

AGENT_NAME = "research_agent"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from agents import Agent, Runner

from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url={BASE_URL}, api_key=os.environ["LITELLM_API_KEY"])
model = OpenAIChatCompletionsModel(model={MODEL}, openai_client=client)

agent = Agent(name=AGENT_NAME, model=model)
result = Runner.run_sync(agent, "What is an agent trace?")
print(result.final_output)`,
    fileName: "agent.py",
  },
  {
    id: "claude",
    label: "Claude Agent SDK",
    logo: anthropicLogo.src,
    install:
      "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http claude-agent-sdk openinference-instrumentation-claude-agent-sdk",
    language: "python",
    quickstart: `import asyncio
import os

AGENT_NAME = "research_agent"

os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"gen_ai.agent.name={AGENT_NAME}"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

options = ClaudeAgentOptions(
    model={MODEL},
    env={"ANTHROPIC_BASE_URL": {PROXY_URL}, "ANTHROPIC_AUTH_TOKEN": os.environ["LITELLM_API_KEY"]},
    tools=[],
    max_turns=1,
)

async def main():
    async for message in query(prompt="What is an agent trace?", options=options):
        if isinstance(message, ResultMessage):
            print(message.result)

asyncio.run(main())`,
    fileName: "agent.py",
    note: "This captures SDK input and output; internal model calls are not exposed by this instrumentor.",
  },
  {
    id: "crewai",
    label: "CrewAI",
    logo: crewaiLogo.src,
    install:
      "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http crewai openinference-instrumentation-crewai",
    language: "python",
    quickstart: `import os

AGENT_NAME = "research_agent"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from crewai import Agent, Crew, Task

from crewai import LLM

model = LLM(
    model={OPENAI_MODEL},
    base_url={BASE_URL},
    api_key=os.environ["LITELLM_API_KEY"],
)

agent = Agent(
    role=AGENT_NAME,
    goal="Answer questions clearly",
    backstory="You explain technical concepts.",
    llm=model,
)
task = Task(description="What is an agent trace?", expected_output="A short answer", agent=agent)
print(Crew(agents=[agent], tasks=[task]).kickoff())`,
    fileName: "agent.py",
  },
  {
    id: "pydantic-ai",
    label: "Pydantic AI",
    logo: pydanticAiLogo.src,
    install: "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http pydantic-ai",
    language: "python",
    quickstart: `import os

AGENT_NAME = "research_agent"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from pydantic_ai import Agent

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    {MODEL},
    provider=OpenAIProvider(base_url={BASE_URL}, api_key=os.environ["LITELLM_API_KEY"]),
)

Agent.instrument_all()
agent = Agent(model, name=AGENT_NAME)
print(agent.run_sync("What is an agent trace?").output)`,
    fileName: "agent.py",
  },
  {
    id: "llamaindex",
    label: "LlamaIndex",
    logo: llamaindexLogo.src,
    install:
      "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http llama-index-core openinference-instrumentation-llama-index llama-index-llms-openai-like",
    language: "python",
    quickstart: `import asyncio
import os

AGENT_NAME = "research_agent"


os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"gen_ai.agent.name={AGENT_NAME}"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument()
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.llms.openai_like import OpenAILike

model = OpenAILike(
    model={MODEL},
    api_base={BASE_URL},
    api_key=os.environ["LITELLM_API_KEY"],
    is_chat_model=True,
    is_function_calling_model=True,
)

agent = FunctionAgent(name=AGENT_NAME, llm=model, tools=[])
async def main():
    result = await agent.run(user_msg="What is an agent trace?")
    print(result)

asyncio.run(main())`,
    fileName: "agent.py",
    note: "The resource attribute supplies the agent name because this instrumentor does not export FunctionAgent.name.",
  },
  {
    id: "adk",
    label: "Google ADK",
    logo: adkLogo.src,
    install: "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http google-adk",
    language: "python",
    quickstart: `import asyncio
import os

AGENT_NAME = "research_agent"

os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "SPAN_ONLY"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(
    model={OPENAI_MODEL},
    api_base={BASE_URL},
    api_key=os.environ["LITELLM_API_KEY"],
)

agent = Agent(name=AGENT_NAME, model=model)
asyncio.run(InMemoryRunner(agent=agent).run_debug("What is an agent trace?"))`,
    fileName: "agent.py",
    note: "SPAN_ONLY records the messages needed to inspect and investigate the run.",
  },
  {
    id: "strands",
    label: "Strands",
    logo: strandsLogo.src,
    install: 'pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http "strands-agents[otel]" openai',
    language: "python",
    quickstart: `import os

AGENT_NAME = "research_agent"


os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental,gen_ai_span_attributes_only"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from strands import Agent

from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={"base_url": {BASE_URL}, "api_key": os.environ["LITELLM_API_KEY"]},
    model_id={MODEL},
)

agent = Agent(name=AGENT_NAME, model=model)
print(agent("What is an agent trace?"))`,
    fileName: "agent.py",
    note: "The semantic-convention setting enables message content in spans.",
  },
  {
    id: "vercel",
    label: "Vercel AI SDK",
    logo: vercelLogo.src,
    install:
      "npm install ai @ai-sdk/otel @opentelemetry/sdk-node @opentelemetry/exporter-trace-otlp-http @ai-sdk/openai-compatible",
    language: "typescript",
    quickstart: `import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { OpenTelemetry } from "@ai-sdk/otel";
import { generateText, registerTelemetry } from "ai";

const sdk = new NodeSDK({ traceExporter: new OTLPTraceExporter() });
sdk.start();
registerTelemetry(new OpenTelemetry());

const AGENT_NAME = "research_agent";
const litellm = createOpenAICompatible({
  name: "litellm",
  baseURL: {BASE_URL},
  apiKey: process.env.LITELLM_API_KEY,
});
const model = litellm({MODEL});

try {
  const { text } = await generateText({
    model,
    prompt: "What is an agent trace?",
    telemetry: { functionId: AGENT_NAME },
  });
  console.log(text);
} finally {
  await sdk.shutdown();
}`,
    fileName: "agent.mts",
  },
  {
    id: "openclaw",
    label: "OpenClaw",
    logo: openclawLogo.src,
    language: "json",
    quickstart: `{
  "agents": {
    "list": [{ "id": "research_agent" }]
  },
  "plugins": {
    "entries": { "diagnostics-otel": { "enabled": true } }
  },
  "diagnostics": {
    "enabled": true,
    "otel": {
      "enabled": true,
      "tracesEndpoint": "{TRACE_URL}",
      "headers": { "Authorization": "Bearer \${LITELLM_API_KEY}" },
      "captureContent": true,
      "traces": true,
      "metrics": false,
      "logs": false,
      "sampleRate": 1
    }
  }
}`,
    existingModel: true,
    fileName: "openclaw.json",
    note: 'Set LITELLM_API_KEY to your LiteLLM key, then run openclaw agent --local --session-id first-trace --message "What is an agent trace?". Select research_agent in Lens. Restart an existing gateway after changing the config.',
    plugin: {
      label: "diagnostics-otel plugin",
      url: "https://docs.openclaw.ai/plugins/reference/diagnostics-otel",
      instruction:
        "Set your agent ID once in ~/.openclaw/openclaw.json. Keep your existing model and workspace settings when adding the tracing configuration.",
    },
  },
  {
    id: "hermes",
    label: "Hermes",
    logo: hermesLogo.src,
    language: "yaml",
    quickstart: `resource_attributes:
  gen_ai.agent.name: research_agent
content_capture: full
backends:
  - type: otlp
    endpoint: {TRACE_URL}
    headers:
      Authorization: "Bearer \${LITELLM_API_KEY}"
    metrics: false
    logs: false`,
    existingModel: true,
    fileName: "hermes_otel.yaml",
    note: "Start a new Hermes session and ask a question. The configured name research_agent appears in Lens. Hermes' built-in diagnostic telemetry alone does not include the conversation content needed for investigations.",
    plugin: {
      label: "community hermes-otel plugin",
      url: "https://github.com/briancaffey/hermes-otel#install",
      instruction: "Set LITELLM_API_KEY to your LiteLLM key, then add this to ~/.hermes/hermes_otel.yaml.",
    },
  },
  {
    id: "otel",
    label: "OpenTelemetry",
    logo: otelLogo.src,
    install: "pip install opentelemetry-distro opentelemetry-exporter-otlp-proto-http openai",
    language: "python",
    quickstart: `import os

AGENT_NAME = "research_agent"

from opentelemetry.instrumentation.auto_instrumentation import initialize

initialize()

from opentelemetry import trace
from openai import OpenAI

client = OpenAI(base_url={BASE_URL}, api_key=os.environ["LITELLM_API_KEY"])

with trace.get_tracer(__name__).start_as_current_span(AGENT_NAME) as span:
    span.set_attribute("gen_ai.agent.name", AGENT_NAME)
    span.set_attribute("openinference.span.kind", "AGENT")
    span.set_attribute("input.value", "What is an agent trace?")
    result = client.chat.completions.create(
        model={MODEL},
        messages=[{"role": "user", "content": "What is an agent trace?"}],
    )
    answer = result.choices[0].message.content
    span.set_attribute("output.value", str(answer))
    print(answer)`,
    fileName: "agent.py",
  },
];

export function frameworkSnippet(guide: FrameworkGuide, proxyUrl: string, model: string, tracingKey = false): string {
  const values: Record<string, string> = {
    MODEL: JSON.stringify(model),
    OPENAI_MODEL: JSON.stringify(`openai/${model}`),
    BASE_URL: JSON.stringify(`${proxyUrl}/v1`),
    PROXY_URL: JSON.stringify(proxyUrl),
    TRACE_URL: `${proxyUrl}/v1/traces`,
  };
  const code = guide.quickstart.replace(
    /\{(MODEL|OPENAI_MODEL|BASE_URL|PROXY_URL|TRACE_URL)\}/g,
    (_, name: string) => values[name],
  );
  return tracingKey && guide.existingModel ? code.replaceAll("${LITELLM_API_KEY}", "${LITELLM_TRACING_KEY}") : code;
}
