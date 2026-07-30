import { getSemanticRouterError, SemanticRouterConfig } from "./build_semantic_router_validation";

const validRouterConfig: SemanticRouterConfig = {
  routes: [{ name: "gpt-4o", description: "general chat", utterances: ["hello there"] }],
};

describe("getSemanticRouterError", () => {
  it("requires an embedding model once the default model and routes are configured", () => {
    expect(
      getSemanticRouterError({
        defaultModel: "gpt-4o",
        embeddingModel: undefined,
        routerConfig: validRouterConfig,
      }),
    ).toBe("Please select an Embedding Model");
  });

  it("treats an empty embedding model string as missing", () => {
    expect(
      getSemanticRouterError({
        defaultModel: "gpt-4o",
        embeddingModel: "",
        routerConfig: validRouterConfig,
      }),
    ).toBe("Please select an Embedding Model");
  });

  it("passes when an embedding model is selected", () => {
    expect(
      getSemanticRouterError({
        defaultModel: "gpt-4o",
        embeddingModel: "text-embedding-3-large",
        routerConfig: validRouterConfig,
      }),
    ).toBeNull();
  });

  it("flags a missing default model before checking the embedding model", () => {
    expect(
      getSemanticRouterError({
        defaultModel: undefined,
        embeddingModel: undefined,
        routerConfig: validRouterConfig,
      }),
    ).toBe("Please select a Default Model");
  });

  it("flags missing routes before checking the embedding model", () => {
    expect(
      getSemanticRouterError({
        defaultModel: "gpt-4o",
        embeddingModel: undefined,
        routerConfig: { routes: [] },
      }),
    ).toBe("Please configure at least one route for the auto router");
  });

  it("validates route completeness after the embedding model is set", () => {
    expect(
      getSemanticRouterError({
        defaultModel: "gpt-4o",
        embeddingModel: "text-embedding-3-large",
        routerConfig: { routes: [{ name: "gpt-4o", description: "", utterances: [] }] },
      }),
    ).toBe("Please ensure all routes have a target model, description, and at least one utterance");
  });
});
