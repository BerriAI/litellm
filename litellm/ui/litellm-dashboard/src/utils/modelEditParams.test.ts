import { describe, expect, it } from "vitest";

import { editedModelInfoPricing } from "./modelEditParams";

const loadedModelInfo = {
  id: "123",
  key: "azure/gpt-5.4",
  mode: "chat",
  input_cost_per_token: 1,
  input_cost_per_token_above_272k_tokens: 2,
  output_cost_per_token_above_272k_tokens: 3,
  cost_per_ptu_per_hour: 4,
};

describe("editedModelInfoPricing", () => {
  it("returns only the rates that differ from what the form loaded", () => {
    expect(
      editedModelInfoPricing(
        { ...loadedModelInfo, input_cost_per_token_above_272k_tokens: 20 },
        loadedModelInfo,
        {},
        {},
      ),
    ).toEqual({ input_cost_per_token_above_272k_tokens: 20 });
  });

  it("returns nothing when the displayed rates were left alone", () => {
    expect(editedModelInfoPricing({ ...loadedModelInfo, mode: "responses" }, loadedModelInfo, {}, {})).toEqual({});
  });

  it("leaves model_info fields that are not per-deployment rates on model_info", () => {
    expect(
      editedModelInfoPricing({ ...loadedModelInfo, cost_per_ptu_per_hour: 40, max_tokens: 5 }, loadedModelInfo, {}, {}),
    ).toEqual({});
  });

  it("drops a non-numeric value typed under a rate key", () => {
    expect(
      editedModelInfoPricing(
        { ...loadedModelInfo, input_cost_per_token_above_272k_tokens: "os.environ/LITELLM_MASTER_KEY" },
        loadedModelInfo,
        {},
        {},
      ),
    ).toEqual({});
  });

  it("lets a rate changed in the LiteLLM Params box win over the same rate edited in Model Info", () => {
    expect(
      editedModelInfoPricing(
        { ...loadedModelInfo, input_cost_per_token_above_272k_tokens: 20, output_cost_per_token_above_272k_tokens: 30 },
        loadedModelInfo,
        { input_cost_per_token_above_272k_tokens: 50, output_cost_per_token_above_272k_tokens: 7 },
        { input_cost_per_token_above_272k_tokens: 6, output_cost_per_token_above_272k_tokens: 7 },
      ),
    ).toEqual({ output_cost_per_token_above_272k_tokens: 30 });
  });

  it("keeps a Model Info edit when the LiteLLM Params box no longer lists a stored override", () => {
    expect(
      editedModelInfoPricing(
        { ...loadedModelInfo, input_cost_per_token_above_272k_tokens: 20 },
        loadedModelInfo,
        {},
        { input_cost_per_token_above_272k_tokens: 6 },
      ),
    ).toEqual({ input_cost_per_token_above_272k_tokens: 20 });
  });
});
