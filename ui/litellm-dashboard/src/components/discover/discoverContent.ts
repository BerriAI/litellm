export const LAUNCH_KINDS = ["model", "provider", "feature"] as const;
export type LaunchKind = (typeof LAUNCH_KINDS)[number];
export type LaunchFilter = LaunchKind | "all";

export interface Launch {
  kind: LaunchKind;
  title: string;
  description: string;
  href: string;
  publishedOn: string;
}

export const LAUNCHES: readonly Launch[] = [
  {
    kind: "feature",
    title: "Errors tab in Usage",
    description: "Failure rate, status code and caller breakdowns for failed requests.",
    href: "https://docs.litellm.ai/release_notes",
    publishedOn: "2026-10-09",
  },
  {
    kind: "provider",
    title: "Microsoft 365 Copilot",
    description: "Chat provider with OAuth token exchange.",
    href: "https://docs.litellm.ai/docs/providers/microsoft_365_copilot",
    publishedOn: "2026-10-09",
  },
  {
    kind: "feature",
    title: "Cmd+K command palette",
    description: "Jump to any page or search keys from anywhere in the UI.",
    href: "https://docs.litellm.ai/release_notes",
    publishedOn: "2026-10-08",
  },
  {
    kind: "model",
    title: "Claude Haiku 5.5",
    description: "Anthropic's newest small model, day 0 support with pricing.",
    href: "https://docs.litellm.ai/blog/claude-haiku-5-5",
    publishedOn: "2026-10-07",
  },
  {
    kind: "feature",
    title: "Decisions API",
    description: "Call decision models at /v1/decisions or /v1/systemone and try them in the Playground.",
    href: "https://docs.litellm.ai/docs/decisions",
    publishedOn: "2026-10-07",
  },
  {
    kind: "feature",
    title: "OpenAI ultrafast tier",
    description: "service_tier: ultrafast on GPT-6 Astra and GPT-6.1 Sol, including /ultrafast in Codex.",
    href: "https://docs.litellm.ai/docs/providers/openai/ultrafast",
    publishedOn: "2026-10-06",
  },
  {
    kind: "model",
    title: "GPT-6.1 Sol",
    description: "OpenAI's GPT-6.1 Sol on OpenAI, Azure and Bedrock.",
    href: "https://docs.litellm.ai/blog/gpt_6_1_sol",
    publishedOn: "2026-09-29",
  },
  {
    kind: "provider",
    title: "TypeSafe Jev",
    description: "New provider with chat, responses and decisions support.",
    href: "https://docs.litellm.ai/blog/typesafe_jev",
    publishedOn: "2026-09-20",
  },
];

export const KIND_LABEL: Record<LaunchKind, string> = {
  model: "New model",
  provider: "New provider",
  feature: "New feature",
};

export const visibleLaunches = (launches: readonly Launch[], filter: LaunchFilter): Launch[] =>
  [...launches]
    .filter((launch) => filter === "all" || launch.kind === filter)
    .sort((a, b) => b.publishedOn.localeCompare(a.publishedOn));

export const FEATURED_MODELS: readonly string[] = [
  "gpt-6.1-sol",
  "claude-haiku-5-5",
  "claude-sonnet-5-5",
  "gemini/gemini-3.8-flash",
  "typesafe/jev-latest",
  "xai/grok-4.6",
  "mistral/mistral-large-4",
  "gpt-6-astra",
];

export const FEATURED_PROVIDERS: readonly string[] = [
  "openai",
  "anthropic",
  "gemini",
  "bedrock",
  "azure",
  "vertex_ai",
  "typesafe",
  "xai",
];

export interface CatalogModel {
  name: string;
  provider: string;
  mode: string;
  contextWindow: number | null;
  inputCostPerToken: number | null;
  outputCostPerToken: number | null;
}

const asNumber = (value: unknown): number | null => (typeof value === "number" ? value : null);

export const toCatalog = (costMap: Record<string, unknown> | undefined): CatalogModel[] =>
  Object.entries(costMap ?? {}).flatMap(([name, raw]) => {
    if (name === "sample_spec" || name.includes("*") || typeof raw !== "object" || raw === null) return [];
    const info = raw as Record<string, unknown>;
    if (typeof info.litellm_provider !== "string") return [];
    return [
      {
        name,
        provider: info.litellm_provider,
        mode: typeof info.mode === "string" ? info.mode : "chat",
        contextWindow: asNumber(info.max_input_tokens) ?? asNumber(info.max_tokens),
        inputCostPerToken: asNumber(info.input_cost_per_token),
        outputCostPerToken: asNumber(info.output_cost_per_token),
      },
    ];
  });

const CHAT_MODES = new Set(["chat", "responses"]);
export const BROWSE_PAGE_SIZE = 12;

export interface BrowseFilter {
  query: string;
  provider: string | null;
}

export const browseModels = (catalog: readonly CatalogModel[], { query, provider }: BrowseFilter): CatalogModel[] => {
  const q = query.trim().toLowerCase();
  if (!q && !provider) {
    const byName = new Map(catalog.map((m) => [m.name, m]));
    return FEATURED_MODELS.flatMap((name) => byName.get(name) ?? []);
  }
  return catalog
    .filter((m) => (provider ? m.provider === provider : true))
    .filter((m) => (q ? m.name.toLowerCase().includes(q) : CHAT_MODES.has(m.mode)))
    .sort((a, b) => (b.contextWindow ?? 0) - (a.contextWindow ?? 0))
    .slice(0, BROWSE_PAGE_SIZE);
};

export const availableProviders = (catalog: readonly CatalogModel[]): string[] => {
  const present = new Set(catalog.map((m) => m.provider));
  return FEATURED_PROVIDERS.filter((p) => present.has(p));
};

export const formatPublishedOn = (isoDate: string): string =>
  new Date(`${isoDate}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

export const formatPerMillion = (costPerToken: number | null): string =>
  costPerToken === null ? "-" : `$${(costPerToken * 1_000_000).toFixed(2)}`;

export const formatTokens = (tokens: number | null): string => {
  if (tokens === null) return "-";
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  return `${Math.round(tokens / 1000)}K`;
};
