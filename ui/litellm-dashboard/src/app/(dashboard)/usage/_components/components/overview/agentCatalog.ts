import anthropicLogo from "../../../../../../../public/assets/logos/anthropic.svg";
import cursorLogo from "../../../../../../../public/assets/logos/cursor.svg";
import copilotLogo from "../../../../../../../public/assets/logos/github_copilot.svg";
import githubLogo from "../../../../../../../public/assets/logos/github.svg";
import hermesLogo from "../../../../../../../public/assets/logos/hermes.png";
import langchainLogo from "../../../../../../../public/assets/logos/langchain.svg";
import litellmLogo from "../../../../../../../public/assets/logos/litellm_monogram.svg";
import openaiLogo from "../../../../../../../public/assets/logos/openai_small.svg";
import openaiAgentsLogo from "../../../../../../../public/assets/logos/openai-agents.svg";
import opencodeLogo from "../../../../../../../public/assets/moyai/logos/opencode.svg";
import pydanticAiLogo from "../../../../../../../public/assets/logos/pydantic-ai-color.svg";
import xaiLogo from "../../../../../../../public/assets/logos/xai.svg";
import type { components } from "@/lib/http/schema";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";

/** One row of `GET /tag/summary`. */
export type TagSummaryRow = components["schemas"]["TagSummaryMetrics"];

export type AgentKind = "agent" | "sdk";

export interface AgentRow {
  id: string;
  label: string;
  description: string;
  kind: AgentKind;
  logo: string | null;
  spend: number;
  tokens: number;
  requests: number;
  users: number;
  /** The raw `User-Agent: ...` tags this agent was built from, used to filter User Agent Activity. */
  tags: string[];
}

interface AgentFamily {
  id: string;
  label: string;
  description: string;
  kind: AgentKind;
  logo: string | null;
  /** Matched against the user-agent with its `User-Agent: ` prefix removed, case-insensitively. */
  match: RegExp;
}

/**
 * Most specific first. Coding agents beat the SDK they are built on, so `codex_sdk_ts` lands on
 * Codex and only a bare `OpenAI/...` client falls through to the OpenAI SDK. Codex wears OpenAI's
 * mark as in the Lens trace view; Claude Code gets its own spark tile rather than Anthropic's wordmark.
 */
const FAMILIES: readonly AgentFamily[] = [
  {
    id: "claude-code",
    label: "Claude Code",
    description: "Anthropic's coding agent",
    kind: "agent",
    // No Claude asset ships in the repo, so TopAgents draws the spark tile itself.
    logo: null,
    match: /^(claude-cli|claude-code|claude_code|claude-user)\b/i,
  },
  {
    id: "codex",
    label: "Codex",
    description: "OpenAI's coding agent",
    kind: "agent",
    logo: openaiLogo.src,
    match: /^(codex[ _-]|codex$)/i,
  },
  {
    id: "opencode",
    label: "OpenCode",
    description: "Open-source terminal coding agent",
    kind: "agent",
    logo: opencodeLogo.src,
    match: /^opencode\b/i,
  },
  {
    id: "cursor",
    label: "Cursor",
    description: "AI code editor",
    kind: "agent",
    logo: cursorLogo.src,
    match: /^cursor\b/i,
  },
  {
    id: "copilot",
    label: "GitHub Copilot",
    description: "GitHub's coding assistant",
    kind: "agent",
    logo: copilotLogo.src,
    match: /^(github-?copilot|copilot)\b/i,
  },
  {
    id: "hermes",
    label: "Hermes Agent",
    description: "Open-source agent",
    kind: "agent",
    logo: hermesLogo.src,
    match: /^hermes\b/i,
  },
  { id: "grok", label: "Grok", description: "xAI's assistant", kind: "agent", logo: xaiLogo.src, match: /^grok\b/i },
  {
    id: "cline",
    label: "Cline",
    description: "Open-source coding agent",
    kind: "agent",
    logo: null,
    match: /^cline\b/i,
  },
  {
    id: "kilo-code",
    label: "Kilo Code",
    description: "Open-source coding agent",
    kind: "agent",
    logo: null,
    match: /^kilo[- ]?code\b/i,
  },
  {
    id: "aider",
    label: "Aider",
    description: "AI pair programming in the terminal",
    kind: "agent",
    logo: null,
    match: /^aider\b/i,
  },
  {
    id: "openai-agents",
    label: "OpenAI Agents",
    description: "OpenAI Agents SDK",
    kind: "sdk",
    logo: openaiAgentsLogo.src,
    match: /^agents$/i,
  },
  {
    id: "langchain",
    label: "LangChain",
    description: "Agent framework",
    kind: "sdk",
    logo: langchainLogo.src,
    match: /^(langchain|langgraph|deepagents)/i,
  },
  {
    id: "pydantic-ai",
    label: "Pydantic AI",
    description: "Agent framework",
    kind: "sdk",
    logo: pydanticAiLogo.src,
    match: /^pydantic-ai\b/i,
  },
  {
    id: "github-actions",
    label: "GitHub Actions",
    description: "CI workflows",
    kind: "sdk",
    logo: githubLogo.src,
    match: /^github-actions\b/i,
  },
  {
    id: "litellm",
    label: "LiteLLM SDK",
    description: "litellm Python SDK",
    kind: "sdk",
    logo: litellmLogo.src,
    match: /^litellm\b/i,
  },
  {
    id: "openai-sdk",
    label: "OpenAI SDK",
    description: "OpenAI client library",
    kind: "sdk",
    logo: openaiLogo.src,
    match: /^(openai|asyncopenai)\b/i,
  },
  {
    id: "anthropic-sdk",
    label: "Anthropic SDK",
    description: "Anthropic client library",
    kind: "sdk",
    logo: anthropicLogo.src,
    match: /^anthropic\b/i,
  },
  {
    id: "python",
    label: "Python",
    description: "Scripts and HTTP clients",
    kind: "sdk",
    logo: null,
    match: /^(python|httpx|aiohttp|urllib|requests)/i,
  },
  {
    id: "node",
    label: "Node.js",
    description: "Scripts and HTTP clients",
    kind: "sdk",
    logo: null,
    match: /^(node|bun|deno|axios|undici)\b/i,
  },
  { id: "curl", label: "curl", description: "Command line", kind: "sdk", logo: null, match: /^curl\b/i },
];

const USER_AGENT_PREFIX = /^user[- ]agent:\s*/i;

export const isUserAgentTag = (tag: string): boolean => USER_AGENT_PREFIX.test(tag.trim());

/** The user-agent product with its version and platform detail dropped: `claude-cli/2.1.263 (external)` -> `claude-cli`. */
export const userAgentProduct = (tag: string): string => {
  const ua = tag.trim().replace(USER_AGENT_PREFIX, "");
  return ua.split(/[/(;]/, 1)[0].trim();
};

const familyFor = (product: string): AgentFamily | null =>
  FAMILIES.find((family) => family.match.test(product)) ?? null;

const BUILDER_AGENT_LABELS: Readonly<Record<string, string>> = {
  browser: "Browser",
  unlabeled: "Unlabeled",
};

const BUILDER_AGENT_DESCRIPTIONS: Readonly<Record<string, string>> = {
  browser: "Browser traffic",
  unlabeled: "Requests without a User-Agent",
};

export const agentRowFor = (id: string): AgentRow => {
  const family = FAMILIES.find((candidate) => candidate.id === id);
  return {
    id,
    label: family?.label ?? BUILDER_AGENT_LABELS[id] ?? id,
    description: family?.description ?? BUILDER_AGENT_DESCRIPTIONS[id] ?? "Custom client",
    kind: family?.kind ?? "sdk",
    logo: family?.logo ?? null,
    spend: 0,
    tokens: 0,
    requests: 0,
    users: 0,
    tags: [],
  };
};

const isBareTag = (row: TagSummaryRow, product: string) =>
  row.tag.trim().replace(USER_AGENT_PREFIX, "").trim() === product;

/**
 * The rows that count toward a product. `/tag/summary` reports a bare tag (`claude-cli`) as the
 * rollup of its versioned children, and a request carries both, so a bare row alone stands for the
 * product. These same rows become the agent's drill-in filter, keeping it from double-counting.
 */
const contributingRows = (product: string, rows: readonly TagSummaryRow[]): readonly TagSummaryRow[] => {
  const bare = rows.find((row) => isBareTag(row, product));
  return bare ? [bare] : rows;
};

const agentFor = (product: string, rows: readonly TagSummaryRow[]): AgentRow => {
  const family = familyFor(product);
  return {
    id: family?.id ?? `ua:${product.toLowerCase()}`,
    label: family?.label ?? product,
    description: family?.description ?? "Custom client",
    kind: family?.kind ?? "sdk",
    logo: family?.logo ?? null,
    spend: rows.reduce((sum, row) => sum + (row.total_spend || 0), 0),
    tokens: rows.reduce((sum, row) => sum + (row.total_tokens || 0), 0),
    requests: rows.reduce((sum, row) => sum + (row.total_requests || 0), 0),
    // Users overlap across a family's products, so the largest is the honest lower bound.
    users: Math.max(0, ...rows.map((row) => row.unique_users || 0)),
    tags: rows.map((row) => row.tag),
  };
};

const mergeAgents = (a: AgentRow, b: AgentRow): AgentRow => ({
  ...a,
  spend: a.spend + b.spend,
  tokens: a.tokens + b.tokens,
  requests: a.requests + b.requests,
  users: Math.max(a.users, b.users),
  tags: [...a.tags, ...b.tags],
});

/**
 * One row per agent from `/tag/summary`. Products are keyed case-sensitively: prod reports `python`
 * and `Python` as distinct products, and folding them let one rollup overwrite the other.
 */
export const topAgents = (rows: readonly TagSummaryRow[]): AgentRow[] => {
  const userAgentRows = rows.filter((row) => isUserAgentTag(row.tag) && userAgentProduct(row.tag));
  const products = [...new Set(userAgentRows.map((row) => userAgentProduct(row.tag)))];
  const perProduct = products.map((product) =>
    agentFor(
      product,
      contributingRows(
        product,
        userAgentRows.filter((row) => userAgentProduct(row.tag) === product),
      ),
    ),
  );
  const byAgent = perProduct.reduce<ReadonlyMap<string, AgentRow>>((acc, agent) => {
    const existing = acc.get(agent.id);
    return new Map([...acc, [agent.id, existing ? mergeAgents(existing, agent) : agent]]);
  }, new Map());
  return [...byAgent.values()]
    .filter((agent) => agent.spend > 0 || agent.tokens > 0 || agent.requests > 0)
    .sort((a, b) => b.tokens - a.tokens || b.spend - a.spend);
};

/**
 * Rewrites one day's `breakdown.entities` (keyed by raw tag) into one entry per agent, keyed by the
 * agent label, so the existing per-day series builders can stack agents the way they stack models.
 * Uses the same bare-vs-versioned rule as `topAgents`, per day, so no day double-counts a product.
 */
const agentEntitiesForDay = (
  entities: Readonly<Record<string, { metrics: SpendMetrics }>>,
): Record<string, { metrics: SpendMetrics }> => {
  const rows: TagSummaryRow[] = Object.entries(entities).map(([tag, { metrics }]) => ({
    tag,
    total_spend: metrics.spend,
    total_tokens: metrics.total_tokens,
    total_requests: metrics.api_requests,
    successful_requests: metrics.successful_requests,
    failed_requests: metrics.failed_requests,
    unique_users: 0,
  }));
  return Object.fromEntries(
    topAgents(rows).map((agent) => [
      agent.label,
      {
        metrics: {
          spend: agent.spend,
          total_tokens: agent.tokens,
          api_requests: agent.requests,
          prompt_tokens: 0,
          completion_tokens: 0,
          successful_requests: 0,
          failed_requests: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
    ]),
  );
};

/**
 * Tag daily activity with each day's raw tags folded into agents, written into `breakdown.models`
 * so `seriesBy(..., "models", ...)` stacks agents per day with no agent-specific chart code.
 */
export const agentDailyData = (days: readonly DailyData[]): DailyData[] =>
  days.map((day) => {
    const models = agentEntitiesForDay(day.breakdown.entities ?? {}) as DailyData["breakdown"]["models"];
    const agentTotal = (key: "spend" | "total_tokens" | "api_requests") =>
      Object.values(models).reduce((sum, entry) => sum + entry.metrics[key], 0);
    // The day total becomes the agents' own sum, so the chart's "Other" means agents past the top N,
    // not the untagged and non-agent traffic in the tag table's day total.
    return {
      ...day,
      metrics: {
        ...day.metrics,
        spend: agentTotal("spend"),
        total_tokens: agentTotal("total_tokens"),
        api_requests: agentTotal("api_requests"),
      },
      breakdown: { ...day.breakdown, models },
    };
  });
