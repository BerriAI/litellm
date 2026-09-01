import { buildAutoRouterTestTargets } from "./build_auto_router_test_targets";

const tierEntries = (
  SIMPLE: string[],
  MEDIUM: string[] = [],
  COMPLEX: string[] = [],
  REASONING: string[] = [],
): [string, string[]][] => [
  ["SIMPLE", SIMPLE],
  ["MEDIUM", MEDIUM],
  ["COMPLEX", COMPLEX],
  ["REASONING", REASONING],
];

const tiers = tierEntries(["gpt-4o-mini"], ["claude-sonnet-4"], ["claude-sonnet-4"], ["o3"]);

describe("buildAutoRouterTestTargets", () => {
  it("dedups tiers that share a model group into one chat target carrying both labels", () => {
    const targets = buildAutoRouterTestTargets({ tiers, semanticMatchingEnabled: false, embeddingModel: undefined });
    expect(targets).toEqual([
      { labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" },
      { labels: ["MEDIUM", "COMPLEX"], modelGroup: "claude-sonnet-4", mode: "chat" },
      { labels: ["REASONING"], modelGroup: "o3", mode: "chat" },
    ]);
  });

  it("emits a target per model when a tier has more than one, and dedups across tiers", () => {
    const targets = buildAutoRouterTestTargets({
      tiers: tierEntries(["gpt-4o-mini", "claude-sonnet-4"], ["claude-sonnet-4"]),
      semanticMatchingEnabled: false,
      embeddingModel: undefined,
    });
    expect(targets).toEqual([
      { labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" },
      { labels: ["SIMPLE", "MEDIUM"], modelGroup: "claude-sonnet-4", mode: "chat" },
    ]);
  });

  it("drops empty/whitespace tiers", () => {
    const targets = buildAutoRouterTestTargets({
      tiers: tierEntries(["gpt-4o-mini"], [], ["   "]),
      semanticMatchingEnabled: false,
      embeddingModel: undefined,
    });
    expect(targets).toEqual([{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]);
  });

  it("returns [] when no tier is configured", () => {
    expect(
      buildAutoRouterTestTargets({
        tiers: tierEntries([]),
        semanticMatchingEnabled: false,
        embeddingModel: undefined,
      }),
    ).toEqual([]);
  });

  it("appends an embedding target only when semantic matching is on and a model is set", () => {
    const targets = buildAutoRouterTestTargets({
      tiers: tierEntries(["gpt-4o-mini"]),
      semanticMatchingEnabled: true,
      embeddingModel: "voyage-3-5",
    });
    expect(targets).toEqual([
      { labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" },
      { labels: ["Embedding"], modelGroup: "voyage-3-5", mode: "embedding" },
    ]);
  });

  it("omits the embedding target when semantic matching is on but no model is chosen", () => {
    const targets = buildAutoRouterTestTargets({
      tiers: tierEntries(["gpt-4o-mini"]),
      semanticMatchingEnabled: true,
      embeddingModel: undefined,
    });
    expect(targets).toEqual([{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]);
  });

  it("omits the embedding target when a model is set but semantic matching is off", () => {
    const targets = buildAutoRouterTestTargets({
      tiers: tierEntries(["gpt-4o-mini"]),
      semanticMatchingEnabled: false,
      embeddingModel: "voyage-3-5",
    });
    expect(targets).toEqual([{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]);
  });

  // A pin outside every tier is still a live fallback destination (empty-tier landings, and a
  // failed LLM classifier routing to it), so a passing test must reach it too or it is only
  // proving the tiers are reachable, not the router.
  it("appends the default model as its own target when it is not already in a tier", () => {
    const targets = buildAutoRouterTestTargets({
      tiers,
      semanticMatchingEnabled: false,
      embeddingModel: undefined,
      defaultModel: "claude-3-opus",
    });
    expect(targets).toEqual([
      { labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" },
      { labels: ["MEDIUM", "COMPLEX"], modelGroup: "claude-sonnet-4", mode: "chat" },
      { labels: ["REASONING"], modelGroup: "o3", mode: "chat" },
      { labels: ["Default"], modelGroup: "claude-3-opus", mode: "chat" },
    ]);
  });

  it("does not duplicate a default model that a tier already covers", () => {
    const targets = buildAutoRouterTestTargets({
      tiers,
      semanticMatchingEnabled: false,
      embeddingModel: undefined,
      defaultModel: "claude-sonnet-4",
    });
    expect(targets).toEqual([
      { labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" },
      { labels: ["MEDIUM", "COMPLEX"], modelGroup: "claude-sonnet-4", mode: "chat" },
      { labels: ["REASONING"], modelGroup: "o3", mode: "chat" },
    ]);
  });

  it.each([[undefined], [""], ["   "]])("adds no default target for %o", (defaultModel) => {
    const targets = buildAutoRouterTestTargets({
      tiers: tierEntries(["gpt-4o-mini"]),
      semanticMatchingEnabled: false,
      embeddingModel: undefined,
      defaultModel,
    });
    expect(targets).toEqual([{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]);
  });
});
