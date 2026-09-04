import { describe, expect, it } from "vitest";
import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { buildAutomaticRouterConfig } from "./auto_setup";

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
  it("uses four different models when four are available", () => {
    const config = buildAutomaticRouterConfig(
      models("expensive", "cheap", "premium", "middle"),
      [deployment("cheap", 1), deployment("middle", 2), deployment("premium", 3), deployment("expensive", 4)],
      {},
    );

    expect(tierModels(config)).toEqual(["cheap", "middle", "premium", "expensive"]);
    expect(config?.classifier_type).toBe("heuristic_v2");
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
      ],
      [deployment("chat-model", 1)],
      {},
    );

    expect(tierModels(config)).toEqual(["chat-model", "chat-model", "chat-model", "chat-model"]);
  });

  it("returns null when there are no usable models", () => {
    expect(buildAutomaticRouterConfig([], [], {})).toBeNull();
  });
});
