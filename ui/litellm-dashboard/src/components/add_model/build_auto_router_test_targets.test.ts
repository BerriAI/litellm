import { buildAutoRouterTestTargets } from "./build_auto_router_test_targets";

const tiers = {
  SIMPLE: "gpt-4o-mini",
  MEDIUM: "claude-sonnet-4",
  COMPLEX: "claude-sonnet-4",
  REASONING: "o3",
};

describe("buildAutoRouterTestTargets", () => {
  it("dedups tiers that share a model group into one chat target carrying both labels", () => {
    const targets = buildAutoRouterTestTargets({ tiers, semanticMatchingEnabled: false, embeddingModel: undefined });
    expect(targets).toEqual([
      { labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" },
      { labels: ["MEDIUM", "COMPLEX"], modelGroup: "claude-sonnet-4", mode: "chat" },
      { labels: ["REASONING"], modelGroup: "o3", mode: "chat" },
    ]);
  });

  it("drops empty/whitespace tiers", () => {
    const targets = buildAutoRouterTestTargets({
      tiers: { SIMPLE: "gpt-4o-mini", MEDIUM: "", COMPLEX: "   ", REASONING: "" },
      semanticMatchingEnabled: false,
      embeddingModel: undefined,
    });
    expect(targets).toEqual([{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]);
  });

  it("returns [] when no tier is configured", () => {
    expect(
      buildAutoRouterTestTargets({
        tiers: { SIMPLE: "", MEDIUM: "", COMPLEX: "", REASONING: "" },
        semanticMatchingEnabled: false,
        embeddingModel: undefined,
      }),
    ).toEqual([]);
  });

  it("appends an embedding target only when semantic matching is on and a model is set", () => {
    const targets = buildAutoRouterTestTargets({
      tiers: { SIMPLE: "gpt-4o-mini", MEDIUM: "", COMPLEX: "", REASONING: "" },
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
      tiers: { SIMPLE: "gpt-4o-mini", MEDIUM: "", COMPLEX: "", REASONING: "" },
      semanticMatchingEnabled: true,
      embeddingModel: undefined,
    });
    expect(targets).toEqual([{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]);
  });

  it("omits the embedding target when a model is set but semantic matching is off", () => {
    const targets = buildAutoRouterTestTargets({
      tiers: { SIMPLE: "gpt-4o-mini", MEDIUM: "", COMPLEX: "", REASONING: "" },
      semanticMatchingEnabled: false,
      embeddingModel: "voyage-3-5",
    });
    expect(targets).toEqual([{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]);
  });
});
