import { describe, expect, it } from "vitest";

import {
  FusionFormValue,
  fusionConfigError,
  fusionModelPayload,
  parseFusionConfig,
  presetFailureMode,
} from "./fusionModelConfig";

const validValue = (overrides: Partial<FusionFormValue> = {}): FusionFormValue => ({
  model_name: "fusion/coding",
  team_id: "",
  panel_models: ["claude", "gpt"],
  aggregator_model: "claude",
  min_successful_panelists: 2,
  panel_timeout_seconds: 120,
  max_candidate_chars: 12000,
  on_quorum_failure: "fail",
  ...overrides,
});

describe("Fusion model configuration", () => {
  it("maps the two user presets to explicit runtime behavior", () => {
    expect(presetFailureMode("quality")).toBe("fail");
    expect(presetFailureMode("resilient")).toBe("aggregator_only");
  });

  it("builds the model/new payload without harness or cross-turn state", () => {
    expect(fusionModelPayload(validValue(), false)).toEqual({
      model_name: "fusion/coding",
      litellm_params: {
        model: "fusion_router",
        fusion_router_config: {
          panel_models: ["claude", "gpt"],
          aggregator_model: "claude",
          min_successful_panelists: 2,
          panel_timeout_seconds: 120,
          max_candidate_chars: 12000,
          on_quorum_failure: "fail",
        },
      },
      model_info: {},
    });
  });

  it("includes team scope only when the caller is required to choose one", () => {
    expect(fusionModelPayload(validValue({ team_id: "team-1" }), true).model_info).toEqual({ team_id: "team-1" });
    expect(fusionConfigError(validValue(), true)).toBe("Select a team to continue.");
  });

  it("rejects undersized panels and impossible quorums", () => {
    expect(fusionConfigError(validValue({ panel_models: ["one"] }), false)).toMatch(/at least two/);
    expect(fusionConfigError(validValue({ min_successful_panelists: 3 }), false)).toMatch(/panel size/);
  });

  it("parses stored configs defensively and supplies stable defaults", () => {
    expect(
      parseFusionConfig({
        panel_models: ["a", "a", "b", 4],
        aggregator_model: "judge",
        on_quorum_failure: "aggregator_only",
      }),
    ).toEqual({
      panel_models: ["a", "b"],
      aggregator_model: "judge",
      min_successful_panelists: 2,
      panel_timeout_seconds: 120,
      max_candidate_chars: 12000,
      on_quorum_failure: "aggregator_only",
    });
  });
});
