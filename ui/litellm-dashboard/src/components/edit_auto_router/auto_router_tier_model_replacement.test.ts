import { describe, expect, it } from "vitest";

import { applyTierSetAction } from "../add_model/tier_set_actions";
import { buildUpdatedComplexityRouterConfig, hydrateComplexityRouterConfig } from "./edit_auto_router_modal";

describe("auto-router tier model replacement", () => {
  it("writes the replacement model and drops params tied to the removed model", () => {
    const stored = {
      tiers: {
        SIMPLE: ["gemma4:12b-it-qat"],
        MEDIUM: [],
        COMPLEX: [],
        REASONING: [],
      },
      tier_model_configs: {
        SIMPLE: [
          {
            model_name: "gemma4:12b-it-qat",
            litellm_params: { reasoning_effort: "medium" },
          },
        ],
      },
      classifier_type: "heuristic" as const,
    };

    const hydrated = hydrateComplexityRouterConfig(stored, "gemma4:12b-it-qat");
    const changed = applyTierSetAction(hydrated, [], {
      kind: "models",
      id: "SIMPLE",
      models: ["gemma4-12b-it-optiq-4bit"],
    }).value;
    const saved = buildUpdatedComplexityRouterConfig(stored, changed);

    expect(saved.tiers).toEqual({
      SIMPLE: ["gemma4-12b-it-optiq-4bit"],
      MEDIUM: [],
      COMPLEX: [],
      REASONING: [],
    });
    expect(saved).not.toHaveProperty("tier_model_configs");
  });
});
