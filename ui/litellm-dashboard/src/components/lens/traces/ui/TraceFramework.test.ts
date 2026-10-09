import { describe, expect, it } from "vitest";
import { traceFramework } from "./TraceFramework";

describe("traceFramework", () => {
  it.each([
    [["claude-agent-sdk", "claude-code"], "Claude Agent SDK", "anthropic"],
    [["deepagents", "langchain_chat_model", "langchain_create_agent"], "Deep Agents", "langchain"],
    [["langchain_chat_model", "langgraph"], "LangGraph", "langgraph"],
    [["langchain_chat_model", "langchain_create_agent"], "LangChain", "langchain"],
    [["crewai", "openai"], "CrewAI", "crewai"],
    [["google-adk"], "Google ADK", "google-adk"],
    [["llama-index"], "LlamaIndex", "llamaindex"],
    [["openai-agents"], "OpenAI Agents", "openai-agents"],
    [["pydantic-ai"], "Pydantic AI", "pydantic-ai"],
    [["strands"], "Strands", "strands"],
    [["vercel-ai-sdk"], "Vercel AI SDK", "vercel"],
    [["openai"], "OpenAI SDK", "openai"],
  ])("resolves %j to %s with the agent framework outranking the model SDK", (frameworks, label, logo) => {
    const framework = traceFramework({ frameworks });
    expect(framework?.label).toBe(label);
    expect(framework?.logo).toContain(logo);
  });

  it("returns null for an unknown or empty framework list", () => {
    expect(traceFramework({ frameworks: ["some-other-sdk"] })).toBeNull();
    expect(traceFramework({ frameworks: [] })).toBeNull();
  });
});
