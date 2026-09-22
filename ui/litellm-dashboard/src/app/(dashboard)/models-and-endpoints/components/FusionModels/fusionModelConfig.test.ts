import { describe, expect, it } from "vitest";

import {
  FusionFormValue,
  fusionConfigError,
  fusionModelPayload,
  parseFusionConfig,
  presetInvocation,
} from "./fusionModelConfig";

const validValue = (overrides: Partial<FusionFormValue> = {}): FusionFormValue => ({
  model_name: "fusion/coding",
  team_id: "",
  outer_model: "outer",
  panel_models: ["panel-a", "panel-b"],
  analyst_model: "analyst",
  invocation: "auto",
  panel_timeout_seconds: 120,
  max_candidate_chars: 12000,
  max_completion_tokens: 16000,
  temperature: 0,
  reasoning_effort: "none",
  search_tool_name: "",
  max_tool_calls: 4,
  web_access_enabled: false,
  ...overrides,
});

describe("Fusion model configuration", () => {
  it("maps the simple presets to invocation behavior", () => {
    expect(presetInvocation("auto")).toBe("auto");
    expect(presetInvocation("always")).toBe("required");
  });

  it("builds the virtual model payload", () => {
    expect(fusionModelPayload(validValue(), false)).toEqual({
      model_name: "fusion/coding",
      litellm_params: {
        model: "fusion_router",
        fusion_router_config: {
          outer_model: "outer",
          panel_models: ["panel-a", "panel-b"],
          analyst_model: "analyst",
          invocation: "auto",
          panel_timeout_seconds: 120,
          max_candidate_chars: 12000,
          max_completion_tokens: 16000,
          temperature: 0,
          reasoning_effort: "none",
          max_tool_calls: 4,
        },
      },
      model_info: {},
    });
  });

  it("omits an analyst to mean same as outer and validates the panel", () => {
    expect(
      fusionModelPayload(validValue({ analyst_model: "" }), false).litellm_params.fusion_router_config,
    ).not.toHaveProperty("analyst_model");
    expect(fusionConfigError(validValue({ panel_models: [] }), false)).toMatch(/at least one/);
    expect(
      fusionConfigError(validValue({ panel_models: Array.from({ length: 9 }, (_, i) => `p-${i}`) }), false),
    ).toMatch(/at most eight/);
  });

  it("requires a Search Tool when web access is enabled", () => {
    expect(fusionConfigError(validValue({ web_access_enabled: true }), false)).toBe(
      "Select a Search Tool or turn Web access off.",
    );
    expect(
      fusionModelPayload(validValue({ web_access_enabled: true, search_tool_name: "web-search" }), false).litellm_params
        .fusion_router_config,
    ).toMatchObject({ search_tool_name: "web-search", max_tool_calls: 4 });
  });

  it("rejects non-finite and fractional numeric settings before sending them", () => {
    expect(fusionConfigError(validValue({ panel_timeout_seconds: Number.NaN }), false)).toMatch(/timeout/);
    expect(fusionConfigError(validValue({ max_candidate_chars: 1000.5 }), false)).toMatch(/Candidate limit/);
    expect(fusionConfigError(validValue({ temperature: Number.NaN }), false)).toMatch(/temperature/);
    expect(fusionConfigError(validValue({ web_access_enabled: true, search_tool_name: "   " }), false)).toMatch(
      /Search Tool/,
    );
  });

  it("includes team scope only when required", () => {
    expect(fusionModelPayload(validValue({ team_id: "team-1" }), true).model_info).toEqual({ team_id: "team-1" });
    expect(fusionConfigError(validValue(), true)).toBe("Select a team to continue.");
  });

  it("parses stored configs defensively", () => {
    const storedConfig = {
      outer_model: "outer",
      panel_models: ["a", "a", "b", 4],
      invocation: "required",
      reasoning_effort: "low",
    };
    const expectedConfig = {
      outer_model: "outer",
      panel_models: ["a", "a", "b"],
      analyst_model: "",
      invocation: "required",
      panel_timeout_seconds: 120,
      max_candidate_chars: 12000,
      max_completion_tokens: 16000,
      temperature: 0,
      reasoning_effort: "low",
      search_tool_name: "",
      max_tool_calls: 4,
    };

    expect(parseFusionConfig(storedConfig)).toEqual(expectedConfig);
  });
});
