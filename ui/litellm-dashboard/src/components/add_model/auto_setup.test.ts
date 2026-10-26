import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import {
  deployedRouterModels,
  recommendSolverPair,
  solverPairConfig,
  nextRouterName,
  simpleSetupError,
} from "./auto_setup";

const deployment = (name: string, model: string, extra: Partial<AutoRouterDeployment> = {}): AutoRouterDeployment => ({
  model_name: name,
  litellm_params: { model },
  ...extra,
});
const resolve = (...rows: AutoRouterDeployment[]) =>
  deployedRouterModels(
    rows.map((row) => ({ model_group: row.model_name!, mode: "chat" })),
    rows,
  );
const names = (pair: ReturnType<typeof recommendSolverPair>) => pair && [pair.everyday.name, pair.hard.name];
const ladder = () =>
  resolve(
    deployment("small", "anthropic/claude-haiku-5-5"),
    deployment("medium", "anthropic/claude-sonnet-5-5"),
    deployment("large", "anthropic/claude-opus-5-5"),
  );

const catalog: Record<string, { mode?: string; litellm_provider?: string }> = JSON.parse(
  readFileSync(new URL("../../../../../model_prices_and_context_window.json", import.meta.url), "utf8"),
);
const catalogOpusModels = Object.entries(catalog).filter(
  ([id, metadata]) => metadata.mode === "chat" && /claude-opus-\d/.test(id) && !/-(?:fast|think)$/.test(id),
);

describe("deployedRouterModels", () => {
  it.each(catalogOpusModels)(
    "recognizes the catalog's standard Opus deployment %s behind a custom alias",
    (id, metadata) => {
      const row = {
        model_name: "custom-deployment",
        litellm_params: { model: id, custom_llm_provider: metadata.litellm_provider },
      };
      const models = deployedRouterModels([{ model_group: "custom-deployment", mode: "chat" }], [row]);
      expect(models).toEqual([expect.objectContaining({ name: "custom-deployment", family: "claude", line: "opus" })]);
    },
  );

  it("resolves underlying identities independently of deployment aliases and serving providers", () => {
    const models = resolve(
      deployment("everyday", "bedrock/us.openai.gpt-6.1-sol"),
      deployment("hard", "bedrock_converse/global.openai.gpt-6-astra"),
    );
    const expectedEveryday = { name: "everyday", family: "openai", line: "sol", version: [6, 1], provider: "Bedrock" };
    const expectedHard = { name: "hard", family: "openai", line: "astra", version: [6], provider: "Bedrock" };
    expect(models).toEqual(
      expect.arrayContaining([expect.objectContaining(expectedEveryday), expect.objectContaining(expectedHard)]),
    );
    expect(names(recommendSolverPair(models, "quality", "openai"))).toEqual(["everyday", "hard"]);
  });

  it("recognizes provider-specific Claude names and a declared base model for opaque deployments", () => {
    const models = resolve(
      deployment("small", "vertex_ai/claude-haiku-4-5@20251001"),
      deployment("large", "bedrock/arn:example:inference-profile/us.anthropic.claude-opus-5-5-v1:0"),
      deployment("middle", "azure/private-alias", {
        litellm_params: { model: "azure/private-alias" },
        model_info: { base_model: "gpt-6.1-sol" },
      }),
    );
    expect(models.map(({ name, family, provider }) => ({ name, family, provider }))).toEqual([
      { name: "large", family: "claude", provider: "Bedrock" },
      { name: "middle", family: "openai", provider: "Azure" },
      { name: "small", family: "claude", provider: "Vertex AI" },
    ]);
  });

  it("uses only caller-visible chat models and excludes blocked and recursive targets", () => {
    const rows = [
      deployment("hidden", "anthropic/claude-opus-6"),
      deployment("blocked", "anthropic/claude-opus-5", { model_info: { blocked: true } }),
      deployment("embedding", "anthropic/claude-opus-5"),
      deployment("recursive", "auto_router/complexity_router"),
      deployment("ready", "anthropic/claude-sonnet-5"),
    ];
    expect(
      deployedRouterModels(
        [
          { model_group: "blocked", mode: "chat" },
          { model_group: "embedding", mode: "embedding" },
          { model_group: "recursive" },
          { model_group: "ready", mode: "chat" },
        ],
        rows,
      ).map((model) => model.name),
    ).toEqual(["ready"]);
  });

  it("does not claim a mixed-version or unknown model group is a known version", () => {
    const rows = [
      deployment("mixed", "anthropic/claude-opus-5"),
      deployment("mixed", "anthropic/claude-opus-5-5"),
      deployment("claude-sonnet-5", "openai/custom-model"),
      deployment("wildcard", "openai/*"),
    ];
    const models = deployedRouterModels(
      [
        { model_group: "mixed", mode: "chat" },
        { model_group: "claude-sonnet-5", mode: "chat" },
        { model_group: "wildcard", mode: "chat" },
      ],
      rows,
    );
    expect(models.map((model) => model.name)).toEqual(["claude-sonnet-5", "mixed"]);
    expect(models.every((model) => model.family === undefined)).toBe(true);
    expect(recommendSolverPair(models, "balanced", "auto")).toBeNull();
  });

  it("keeps homogeneous replicas and ignores blocked replicas when resolving identity", () => {
    const rows = [
      deployment("group", "anthropic/claude-opus-5-5"),
      deployment("group", "bedrock/anthropic.claude-opus-5-5-v1:0"),
      deployment("group", "anthropic/claude-opus-5", { model_info: { blocked: true } }),
    ];
    const models = deployedRouterModels([{ model_group: "group" }], rows);
    expect(models).toHaveLength(1);
    expect(models[0]).toMatchObject({ name: "group", version: [5, 5], provider: "Anthropic, Bedrock" });
  });

  it("keeps arbitrary canonical names available without pretending an unversioned alias is latest", () => {
    expect(
      deployedRouterModels([{ model_group: "claude-opus-8-12" }, { model_group: "claude-opus-latest" }], []).map(
        (model) => model.name,
      ),
    ).toEqual(["claude-opus-8-12", "claude-opus-latest"]);
  });
});

describe("recommendSolverPair", () => {
  it.each([
    ["cheapest", ["small", "medium"]],
    ["balanced", ["small", "large"]],
    ["quality", ["medium", "large"]],
  ] as const)("uses the requested %s pair", (goal, expected) => {
    expect(names(recommendSolverPair(ladder(), goal, "claude"))).toEqual(expected);
  });

  it("selects future deployed releases numerically within each line, without a version allowlist", () => {
    const models = resolve(
      deployment("old-small", "anthropic/claude-haiku-8-9"),
      deployment("new-small", "bedrock/anthropic.claude-haiku-8-10-v1:0"),
      deployment("old-large", "anthropic/claude-opus-8-9"),
      deployment("new-large", "anthropic/claude-opus-9"),
    );
    expect(names(recommendSolverPair(models, "balanced", "claude"))).toEqual(["new-small", "new-large"]);
  });

  it("orders snapshots within a version without confusing a snapshot with the next version", () => {
    const models = resolve(
      deployment("old", "anthropic/claude-haiku-4-5-20251001"),
      deployment("snapshot", "anthropic/claude-haiku-4-5-20251101"),
      deployment("next", "anthropic/claude-haiku-4-6"),
      deployment("large", "anthropic/claude-opus-5"),
    );
    expect(names(recommendSolverPair(models, "balanced", "claude"))).toEqual(["next", "large"]);
    expect(
      names(
        recommendSolverPair(
          models.filter((model) => model.name !== "next"),
          "balanced",
          "claude",
        ),
      ),
    ).toEqual(["snapshot", "large"]);
  });

  it("prefers rolling aliases over snapshots of the same release", () => {
    const models = resolve(
      deployment("pin", "anthropic/claude-haiku-4-5-20251101"),
      deployment("alias", "anthropic/claude-haiku-4-5"),
      deployment("hard", "anthropic/claude-opus-5"),
      deployment("luna-pin", "openai/gpt-6-luna-2026-01-01"),
      deployment("luna", "openai/gpt-6-luna"),
      deployment("astra", "openai/gpt-6-astra"),
    );
    expect(names(recommendSolverPair(models, "balanced", "claude"))).toEqual(["alias", "hard"]);
    expect(names(recommendSolverPair(models, "balanced", "openai"))).toEqual(["luna", "astra"]);
  });

  it.each([
    { models: ["grok-4.20", "grok-4.20-0309"], expected: "grok-4.20" },
    { models: ["grok-4.20-0309", "grok-4.20-0410"], expected: "grok-4.20-0410" },
    { models: ["grok-4.20-0309", "grok-4.21"], expected: "grok-4.21" },
  ])("orders Grok releases and short-date snapshots: $expected", ({ models, expected }) => {
    const available = resolve(
      deployment("everyday", "openai/gpt-6-luna"),
      ...models.map((model) => deployment(model, `xai/${model}`)),
    );
    expect(names(recommendSolverPair(available, "balanced", "grok"))).toEqual(["everyday", expected]);
  });

  it("does not rank a major-release snapshot above a newer minor release", () => {
    const models = resolve(
      deployment("old", "anthropic/claude-5-haiku-20260101"),
      deployment("next", "anthropic/claude-haiku-5-1"),
      deployment("hard", "anthropic/claude-opus-5"),
    );
    expect(names(recommendSolverPair(models, "balanced", "claude"))).toEqual(["next", "hard"]);
  });

  it("prefers a same-family substitute over a matching rank from another family", () => {
    const models = [
      ...ladder().filter((model) => model.line !== "haiku"),
      ...resolve(deployment("luna", "openai/gpt-6-luna")),
    ];
    const pair = recommendSolverPair(models, "balanced", "claude");
    expect(names(pair)).toEqual(["medium", "large"]);
    expect(pair?.notice).toContain("closest available pair");
  });

  it("prefers a complete family for Auto and explains a cross-family substitute when explicitly requested", () => {
    const models = [
      ...ladder().filter((model) => model.line === "sonnet"),
      ...resolve(
        deployment("luna", "bedrock/us.openai.gpt-6-luna"),
        deployment("astra", "bedrock/us.openai.gpt-6-astra"),
      ),
    ];
    expect(names(recommendSolverPair(models, "balanced", "auto"))).toEqual(["luna", "astra"]);
    const selected = recommendSolverPair(models, "quality", "claude");
    expect(names(selected)).toEqual(["medium", "astra"]);
    expect(selected?.notice).toContain("two model families");
  });

  it.each([
    ["glm", "zai/glm-7.3-flash", "zai/glm-7.3"],
    ["deepseek", "deepseek/deepseek-v9-flash", "deepseek/deepseek-v9-pro"],
  ] as const)("handles the %s two-line family for every goal", (family, small, large) => {
    const models = resolve(deployment("small", small), deployment("large", large));
    for (const goal of ["cheapest", "balanced", "quality"] as const)
      expect(names(recommendSolverPair(models, goal, family))).toEqual(["small", "large"]);
  });

  it("does not fabricate a second solver or substitute a different family when none of the requested family exists", () => {
    expect(recommendSolverPair(ladder().slice(0, 1), "balanced", "auto")).toBeNull();
    expect(recommendSolverPair(ladder(), "balanced", "openai")).toBeNull();
    expect(recommendSolverPair([], "balanced", "auto")).toBeNull();
  });

  it("retains Gemini family suggestions and recognizes the existing Grok fallback", () => {
    const models = resolve(
      deployment("lite", "gemini/gemini-3.5-flash-lite"),
      deployment("flash-old", "vertex_ai/gemini-3.5-flash"),
      deployment("flash", "vertex_ai/gemini-3.8-flash"),
      deployment("pro", "gemini/gemini-3.1-pro-preview"),
      deployment("grok", "xai/grok-4.6"),
    );
    expect(names(recommendSolverPair(models, "cheapest", "gemini"))).toEqual(["lite", "flash"]);
    expect(names(recommendSolverPair(models, "balanced", "gemini"))).toEqual(["lite", "pro"]);
    expect(names(recommendSolverPair(models, "quality", "gemini"))).toEqual(["flash", "pro"]);
    expect(models.find((m) => m.name === "grok")).toMatchObject({ family: "grok", provider: "xAI" });
    expect(names(recommendSolverPair(models, "balanced", "grok"))).toEqual(["lite", "grok"]);
  });

  it("uses the latest deployed Terra as a fallback without replacing the preferred OpenAI pairs", () => {
    const models = resolve(
      deployment("terra-old", "openai/gpt-5.6-terra"),
      deployment("terra", "openai/gpt-6.1-terra"),
      deployment("sol", "openai/gpt-6.1-sol"),
    );
    expect(names(recommendSolverPair(models, "balanced", "openai"))).toEqual(["terra", "sol"]);
    const complete = [
      ...models,
      ...resolve(deployment("luna", "openai/gpt-6-luna"), deployment("astra", "openai/gpt-6-astra")),
    ];
    expect(names(recommendSolverPair(complete, "cheapest", "openai"))).toEqual(["luna", "sol"]);
    expect(names(recommendSolverPair(complete, "balanced", "openai"))).toEqual(["luna", "astra"]);
    expect(names(recommendSolverPair(complete, "quality", "openai"))).toEqual(["sol", "astra"]);
  });

  it("suggests a deterministic equal-rank cross-family fallback only when no ordered pair exists", () => {
    const models = resolve(deployment("opus", "anthropic/claude-opus-5-5"), deployment("astra", "openai/gpt-6-astra"));
    const pair = recommendSolverPair(models, "balanced", "auto");
    expect(names(pair)).toEqual(["astra", "opus"]);
    expect(pair?.notice).toContain("two model families");
    expect(names(recommendSolverPair([...models].reverse(), "balanced", "auto"))).toEqual(names(pair));
    const withLuna = [...models, ...resolve(deployment("luna", "openai/gpt-6-luna"))];
    expect(names(recommendSolverPair(withLuna, "balanced", "auto"))).toEqual(["luna", "astra"]);
  });

  it("does not prefer native-provider deployments over a newer cloud deployment", () => {
    const models = resolve(
      deployment("native", "openai/gpt-6-luna"),
      deployment("cloud", "bedrock/global.openai.gpt-6.2-luna"),
      deployment("strong", "bedrock/global.openai.gpt-6-astra"),
    );
    expect(names(recommendSolverPair(models, "balanced", "openai"))).toEqual(["cloud", "strong"]);
  });
});

describe("solverPairConfig", () => {
  it("builds two solver references using built-in rules without paid classifiers, custom tiers or output overrides", () => {
    expect(solverPairConfig("everyday", "hard")).toEqual({
      classifier_type: "heuristic",
      classification_mode: "user_turn",
      tiers: { SIMPLE: ["everyday"], MEDIUM: ["everyday"], COMPLEX: ["hard"], REASONING: ["hard"] },
    });
  });
});

describe("simple setup validation", () => {
  it("avoids existing names without requiring a naming decision", () => {
    expect(nextRouterName([])).toBe("auto-router");
    expect(nextRouterName(["auto-router", "auto-router-2", "auto-router-4"])).toBe("auto-router-3");
  });
  it("blocks stale or incomplete inventory and same-line selections", () => {
    const [small, large] = [ladder()[0], ladder()[2]];
    expect(simpleSetupError(true, false, small, large)).not.toBeNull();
    expect(simpleSetupError(false, true, small, large)).not.toBeNull();
    expect(simpleSetupError(false, false, small)).not.toBeNull();
    expect(simpleSetupError(false, false, small, small)).not.toBeNull();
    expect(simpleSetupError(false, false, small, large)).toBeNull();
  });
});
