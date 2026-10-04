import anthropicLogo from "../../../../../public/assets/logos/anthropic.svg";
import crewaiLogo from "../../../../../public/assets/logos/crewai-color.svg";
import cursorLogo from "../../../../../public/assets/logos/cursor.svg";
import copilotLogo from "../../../../../public/assets/logos/github_copilot.svg";
import googleAdkLogo from "../../../../../public/assets/logos/google-adk.png";
import langchainLogo from "../../../../../public/assets/logos/langchain.svg";
import langgraphLogo from "../../../../../public/assets/logos/langgraph-color.svg";
import llamaIndexLogo from "../../../../../public/assets/logos/llamaindex-color.svg";
import openaiLogo from "../../../../../public/assets/logos/openai_small.svg";
import openaiAgentsLogo from "../../../../../public/assets/logos/openai-agents.svg";
import pydanticAiLogo from "../../../../../public/assets/logos/pydantic-ai-color.svg";
import strandsLogo from "../../../../../public/assets/logos/strands.svg";
import vercelLogo from "../../../../../public/assets/logos/vercel.svg";
import { Logo } from "@/components/molecules/logo/Logo";
import { cn } from "@/lib/cva.config";

import type { TraceSummary } from "../types";

export interface TraceFramework {
  readonly id: string;
  readonly label: string;
  readonly logo: string;
}

/**
 * Ids are the slugs the trace normalizer emits (litellm-rust `Integration`, kebab-case) plus the raw
 * `ls_integration` values LangChain reports. A trace lists every framework seen on its spans, so the
 * registry runs most-specific first: an agent framework beats the model SDK it wraps.
 */
const FRAMEWORKS: readonly TraceFramework[] = [
  { id: "claude-agent-sdk", label: "Claude Agent SDK", logo: anthropicLogo.src },
  { id: "claude-code", label: "Claude Code", logo: anthropicLogo.src },
  { id: "openai-codex", label: "Codex", logo: openaiLogo.src },
  { id: "cursor", label: "Cursor", logo: cursorLogo.src },
  { id: "copilot", label: "GitHub Copilot", logo: copilotLogo.src },
  { id: "deepagents", label: "Deep Agents", logo: langchainLogo.src },
  { id: "langgraph", label: "LangGraph", logo: langgraphLogo.src },
  { id: "langchain", label: "LangChain", logo: langchainLogo.src },
  { id: "langchain_create_agent", label: "LangChain", logo: langchainLogo.src },
  { id: "langchain_chat_model", label: "LangChain", logo: langchainLogo.src },
  { id: "crewai", label: "CrewAI", logo: crewaiLogo.src },
  { id: "google-adk", label: "Google ADK", logo: googleAdkLogo.src },
  { id: "llama-index", label: "LlamaIndex", logo: llamaIndexLogo.src },
  { id: "openai-agents", label: "OpenAI Agents", logo: openaiAgentsLogo.src },
  { id: "pydantic-ai", label: "Pydantic AI", logo: pydanticAiLogo.src },
  { id: "strands", label: "Strands", logo: strandsLogo.src },
  { id: "vercel-ai-sdk", label: "Vercel AI SDK", logo: vercelLogo.src },
  { id: "openai", label: "OpenAI SDK", logo: openaiLogo.src },
];

export const traceFramework = (summary: Pick<TraceSummary, "frameworks">): TraceFramework | null =>
  FRAMEWORKS.find((framework) => summary.frameworks?.includes(framework.id)) ?? null;

export function FrameworkLogo({ framework, className }: { framework: TraceFramework; className?: string }) {
  return (
    <span aria-hidden className="contents">
      <Logo src={framework.logo} label={framework.label} className={cn("size-3.5 shrink-0", className)} />
    </span>
  );
}
