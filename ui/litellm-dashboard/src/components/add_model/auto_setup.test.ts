import { describe, expect, it } from "vitest";
import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { buildModelAvailability } from "@/lib/autorouter_presets";
import { buildAutomaticRouterConfig, buildPreferredTierModels, type PreferredTierModels } from "./auto_setup";

const models = (...names: string[]) => names.map((model_group) => ({ model_group, mode: "chat" }));
const reasoningModel = (model_group: string, supported_reasoning_efforts: string[]) => ({
  model_group,
  mode: "chat",
  supports_reasoning: true,
  supported_reasoning_efforts,
});
const deployment = (model_name: string, model = model_name): AutoRouterDeployment => ({
  model_name,
  litellm_params: { model },
});
const tierModels = (config: ReturnType<typeof buildAutomaticRouterConfig>) =>
  config && Object.values(config.tiers).map((tier) => (typeof tier === "string" ? tier : tier[0]));

describe("buildPreferredTierModels", () => {
  it("recognizes curated models that are not in a preset", () => {
    const available = ["gpt-5.6-luna", "claude-sonnet-5", "grok-4.6", "deepseek-v4-pro"];
    const availability = buildModelAvailability(available, []);
    const preferred = buildPreferredTierModels([], availability);
    const expected: PreferredTierModels = {
      SIMPLE: ["gpt-5.6-luna"],
      MEDIUM: ["claude-sonnet-5"],
      COMPLEX: ["deepseek-v4-pro", "grok-4.6"],
      REASONING: ["deepseek-v4-pro", "grok-4.6"],
    };

    expect(preferred).toEqual(expected);
  });

  it("prefers the current model ladder over older preset entries", () => {
    const availability = buildModelAvailability(["gpt-5.6-luna", "gpt-4o-mini"], []);
    const preferred = buildPreferredTierModels(
      [
        {
          key: "old",
          label: "Old",
          description: "Old model",
          complexity_router_config: {
            tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
            classifier_type: "heuristic_v2",
          },
        },
      ],
      availability,
    );

    expect(preferred.SIMPLE).toEqual(["gpt-5.6-luna", "gpt-4o-mini"]);
  });
});

describe("buildAutomaticRouterConfig", () => {
  it("selects one preferred model for each tier", () => {
    const preferred: PreferredTierModels = {
      SIMPLE: ["simple"],
      MEDIUM: ["medium"],
      COMPLEX: ["complex"],
      REASONING: ["reasoning"],
    };

    expect(
      tierModels(buildAutomaticRouterConfig(models("simple", "medium", "complex", "reasoning"), [], preferred)),
    ).toEqual(["simple", "medium", "complex", "reasoning"]);
  });

  it.each([
    {
      provider: "OpenAI",
      available: ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-6-astra"],
      expected: ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-6-astra", "gpt-6-astra"],
      supportedEfforts: ["low", "medium", "high", "xhigh", "max"],
      effort: "max",
    },
    {
      provider: "Anthropic",
      available: ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"],
      expected: ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5", "claude-opus-5"],
      supportedEfforts: ["low", "medium", "high", "max"],
      effort: "max",
    },
    {
      provider: "Google",
      available: ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-pro-preview"],
      expected: ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview"],
      supportedEfforts: ["low", "medium", "high"],
      effort: "high",
    },
    {
      provider: "DeepSeek",
      available: ["deepseek-v4-flash", "deepseek-v4-pro"],
      expected: ["deepseek-v4-flash", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-pro"],
      supportedEfforts: ["none", "high"],
      effort: "high",
    },
    {
      provider: "xAI",
      available: ["grok-4.6"],
      expected: ["grok-4.6", "grok-4.6", "grok-4.6", "grok-4.6"],
      supportedEfforts: ["low", "medium", "high", "xhigh"],
      effort: "xhigh",
    },
  ])(
    "uses the current $provider ladder and strongest advertised reasoning effort",
    ({ available, expected, supportedEfforts, effort }) => {
      const availability = buildModelAvailability(available, []);
      const preferred = buildPreferredTierModels([], availability);
      const modelInfo = models(...available).map((model) =>
        model.model_group === expected[3] ? reasoningModel(model.model_group, supportedEfforts) : model,
      );

      const config = buildAutomaticRouterConfig(modelInfo, [], preferred);

      expect(tierModels(config)).toEqual(expected);
      expect(config?.tier_model_params).toEqual({
        REASONING: { [expected[3]]: { reasoning_effort: effort } },
      });
    },
  );

  it("never exceeds the selected model group's advertised reasoning efforts", () => {
    const available = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"];
    const availability = buildModelAvailability(available, []);
    const preferred = buildPreferredTierModels([], availability);
    const modelInfo = [
      ...models("gpt-5.6-luna", "gpt-5.6-terra"),
      reasoningModel("gpt-5.6-sol", ["none", "low", "medium", "high", "xhigh"]),
    ];

    const config = buildAutomaticRouterConfig(modelInfo, [], preferred);

    expect(config?.tier_model_params).toEqual({
      REASONING: { "gpt-5.6-sol": { reasoning_effort: "xhigh" } },
    });
  });

  it("leaves reasoning effort unset when the proxy does not report supported values", () => {
    const available = ["grok-4.6"];
    const availability = buildModelAvailability(available, []);
    const preferred = buildPreferredTierModels([], availability);

    const config = buildAutomaticRouterConfig(models(...available), [], preferred);

    expect(config?.tier_model_params).toBeUndefined();
  });

  it("reuses the closest available tier when a tier has no match", () => {
    const preferred: PreferredTierModels = {
      SIMPLE: ["simple"],
      MEDIUM: [],
      COMPLEX: ["complex"],
      REASONING: [],
    };

    expect(tierModels(buildAutomaticRouterConfig(models("simple", "complex"), [], preferred))).toEqual([
      "simple",
      "simple",
      "complex",
      "complex",
    ]);
  });

  it("returns null when none of the available models are recommended", () => {
    const preferred: PreferredTierModels = {
      SIMPLE: ["missing-simple"],
      MEDIUM: ["missing-medium"],
      COMPLEX: ["missing-complex"],
      REASONING: ["missing-reasoning"],
    };

    expect(buildAutomaticRouterConfig(models("unknown-model"), [], preferred)).toBeNull();
  });

  it("ignores non-chat models and existing auto routers", () => {
    const preferred: PreferredTierModels = {
      SIMPLE: ["gpt-4o-mini", "smart-router"],
      MEDIUM: [],
      COMPLEX: [],
      REASONING: [],
    };
    const available = [
      { model_group: "gpt-4o-mini", mode: "chat" },
      { model_group: "image-model", mode: "image_generation" },
      { model_group: "smart-router", mode: "chat" },
    ];

    expect(
      tierModels(
        buildAutomaticRouterConfig(available, [deployment("smart-router", "auto_router/complexity_router")], preferred),
      ),
    ).toEqual(["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"]);
  });
});
