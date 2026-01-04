import { transformModelData } from "./modelDataTransformer";
import { describe, it, expect } from "vitest";
describe("transformModelData", () => {
  const mockGetProviderFromModel = (model: string) => {
    if (model.includes("gpt")) return "openai";
    if (model.includes("claude")) return "anthropic";
    return "openai";
  };

  it("should transform raw model data correctly", () => {
    const rawData = {
      data: [
        {
          model_name: "gpt-4",
          litellm_params: {
            model: "gpt-4",
            api_base: "https://api.openai.com",
            api_key: "sk-123",
          },
          model_info: {
            input_cost_per_token: 0.0000015,
            output_cost_per_token: 0.000002,
            max_tokens: 8192,
            max_input_tokens: 128000,
          },
        },
      ],
    };

    const result = transformModelData(rawData, mockGetProviderFromModel);

    expect(result.data[0]).toHaveProperty("provider", "openai");
    expect(result.data[0]).toHaveProperty("input_cost", "1.50");
    expect(result.data[0]).toHaveProperty("output_cost", "2.00");
    expect(result.data[0]).toHaveProperty("max_tokens", 8192);
    expect(result.data[0]).toHaveProperty("max_input_tokens", 128000);
    expect(result.data[0]).toHaveProperty("api_base", "https://api.openai.com");
    expect(result.data[0]).toHaveProperty("litellm_model_name", "gpt-4");
    expect(result.data[0]).toHaveProperty("cleanedLitellmParams");
    expect(result.data[0].cleanedLitellmParams).not.toHaveProperty("model");
    expect(result.data[0].cleanedLitellmParams).not.toHaveProperty("api_base");
  });

  it("should handle empty data", () => {
    const result = transformModelData({ data: [] }, mockGetProviderFromModel);
    expect(result).toEqual({ data: [] });
  });

  it("should handle null/undefined data", () => {
    const result = transformModelData(null, mockGetProviderFromModel);
    expect(result).toEqual({ data: [] });
  });
});
