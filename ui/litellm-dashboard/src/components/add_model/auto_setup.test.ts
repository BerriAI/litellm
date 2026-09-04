import { describe, expect, it } from "vitest";
import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { buildAutomaticRouterConfig, type PreferredTierModels } from "./auto_setup";

const models = (...names: string[]) => names.map((model_group) => ({ model_group, mode: "chat" }));

const deployment = (name: string, cost: number): AutoRouterDeployment => ({
  model_name: name,
  litellm_params: {
    model: name,
    input_cost_per_token: cost / 2,
    output_cost_per_token: cost / 2,
  },
});

const firstModel = (value: string | string[]): string => (typeof value === "string" ? value : value[0]);

const tierModels = (config: ReturnType<typeof buildAutomaticRouterConfig>) =>
  config && [
    firstModel(config.tiers.SIMPLE),
    firstModel(config.tiers.MEDIUM),
    firstModel(config.tiers.COMPLEX),
    firstModel(config.tiers.REASONING),
  ];

describe("buildAutomaticRouterConfig", () => {
  it("uses available preferred models before price ranking", () => {
    const preferred: PreferredTierModels = {
      SIMPLE: ["preferred-simple"],
      MEDIUM: ["preferred-medium"],
      COMPLEX: ["preferred-complex"],
      REASONING: ["preferred-reasoning"],
    };
    const available = [...Object.values(preferred).flat(), "cheap-decoy", "expensive-decoy"];
    const config = buildAutomaticRouterConfig(
      models(...available),
      available.map((name, index) => deployment(name, index + 1)),
      {},
      preferred,
    );

    expect(tierModels(config)).toEqual([
      "preferred-simple",
      "preferred-medium",
      "preferred-complex",
      "preferred-reasoning",
    ]);
  });

  it("reuses the nearest preferred model for tiers with no preferred match", () => {
    const preferred: PreferredTierModels = {
      SIMPLE: ["preferred-simple"],
      MEDIUM: [],
      COMPLEX: ["preferred-complex"],
      REASONING: [],
    };
    const config = buildAutomaticRouterConfig(
      models("preferred-simple", "preferred-complex", "cheap-decoy"),
      [deployment("preferred-simple", 4), deployment("preferred-complex", 5), deployment("cheap-decoy", 1)],
      {},
      preferred,
    );

    expect(tierModels(config)).toEqual([
      "preferred-simple",
      "preferred-simple",
      "preferred-complex",
      "preferred-complex",
    ]);
  });

  it("uses price ranking when none of the preferred models are available", () => {
    const unavailablePreferred: PreferredTierModels = {
      SIMPLE: ["missing-simple"],
      MEDIUM: ["missing-medium"],
      COMPLEX: ["missing-complex"],
      REASONING: ["missing-reasoning"],
    };
    const config = buildAutomaticRouterConfig(
      models("expensive", "cheap", "premium", "middle"),
      [deployment("cheap", 1), deployment("middle", 2), deployment("premium", 3), deployment("expensive", 4)],
      {},
      unavailablePreferred,
    );

    expect(tierModels(config)).toEqual(["cheap", "middle", "premium", "expensive"]);
  });

  it("uses four different models when four are available", () => {
    const config = buildAutomaticRouterConfig(
      models("expensive", "cheap", "premium", "middle"),
      [deployment("cheap", 1), deployment("middle", 2), deployment("premium", 3), deployment("expensive", 4)],
      {},
    );

    expect(tierModels(config)).toEqual(["cheap", "middle", "premium", "expensive"]);
    expect(config?.classifier_type).toBe("heuristic_v2");
  });

  it("selects one model per tier from a large inventory", () => {
    const names = Array.from({ length: 100 }, (_, index) => `model-${index.toString().padStart(3, "0")}`);
    const config = buildAutomaticRouterConfig(
      models(...names),
      names.map((name, index) => deployment(name, index + 1)),
      {},
    );
    const expectedTiers = {
      SIMPLE: ["model-000"],
      MEDIUM: ["model-033"],
      COMPLEX: ["model-066"],
      REASONING: ["model-099"],
    };

    expect(config?.tiers).toEqual(expectedTiers);
  });

  it("only repeats models when fewer than four are available", () => {
    expect(
      tierModels(
        buildAutomaticRouterConfig(
          models("cheap", "expensive"),
          [deployment("cheap", 1), deployment("expensive", 4)],
          {},
        ),
      ),
    ).toEqual(["cheap", "cheap", "expensive", "expensive"]);
  });

  it("uses the published cost map when deployments do not define prices", () => {
    const config = buildAutomaticRouterConfig(
      models("premium", "cheap", "middle"),
      [
        { model_name: "premium", litellm_params: { model: "provider/premium" } },
        { model_name: "cheap", litellm_params: { model: "provider/cheap" } },
        { model_name: "middle", litellm_params: { model: "provider/middle" } },
      ],
      {
        "provider/cheap": { input_cost_per_token: 1, output_cost_per_token: 1 },
        "provider/middle": { input_cost_per_token: 2, output_cost_per_token: 2 },
        "provider/premium": { input_cost_per_token: 3, output_cost_per_token: 3 },
      },
    );

    expect(tierModels(config)).toEqual(["cheap", "middle", "premium", "premium"]);
  });

  it("uses the most expensive deployment when a group has several", () => {
    const config = buildAutomaticRouterConfig(
      models("variable", "steady", "premium", "top"),
      [
        deployment("variable", 1),
        deployment("variable", 8),
        deployment("steady", 2),
        deployment("premium", 3),
        deployment("top", 4),
      ],
      {},
    );

    expect(tierModels(config)).toEqual(["steady", "premium", "top", "variable"]);
  });

  it("ignores non-chat and existing auto-router models", () => {
    const config = buildAutomaticRouterConfig(
      [
        { model_group: "chat-model", mode: "chat" },
        { model_group: "image-model", mode: "image_generation" },
        { model_group: "auto_router/existing", mode: "chat" },
        { model_group: "smart-router", mode: "chat" },
      ],
      [
        deployment("chat-model", 1),
        { model_name: "smart-router", litellm_params: { model: "auto_router/complexity_router" } },
      ],
      {},
    );

    expect(tierModels(config)).toEqual(["chat-model", "chat-model", "chat-model", "chat-model"]);
  });

  it("returns null when there are no usable models", () => {
    expect(buildAutomaticRouterConfig([], [], {})).toBeNull();
  });
});
