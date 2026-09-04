import { describe, expect, it } from "vitest";
import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { buildModelAvailability } from "@/lib/autorouter_presets";
import { buildAutomaticRouterConfig, buildPreferredTierModels, type PreferredTierModels } from "./auto_setup";

const models = (...names: string[]) => names.map((model_group) => ({ model_group, mode: "chat" }));
const deployment = (model_name: string, model = model_name): AutoRouterDeployment => ({
  model_name,
  litellm_params: { model },
});
const tierModels = (config: ReturnType<typeof buildAutomaticRouterConfig>) =>
  config && Object.values(config.tiers).map((tier) => (typeof tier === "string" ? tier : tier[0]));

describe("buildPreferredTierModels", () => {
  it("recognizes curated models that are not in a preset", () => {
    const available = ["gpt-4o-mini", "claude-sonnet-4-5", "grok-4", "deepseek-reasoner"];
    const availability = buildModelAvailability(available, []);
    const preferred = buildPreferredTierModels([], availability);
    const expected: PreferredTierModels = {
      SIMPLE: ["gpt-4o-mini"],
      MEDIUM: ["claude-sonnet-4-5"],
      COMPLEX: ["grok-4"],
      REASONING: ["deepseek-reasoner"],
    };

    expect(preferred).toEqual(expected);
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
