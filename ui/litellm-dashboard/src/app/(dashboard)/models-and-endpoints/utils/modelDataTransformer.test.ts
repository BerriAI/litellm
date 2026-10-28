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

  it("should handle zero cost models correctly", () => {
    const rawData = {
      data: [
        {
          model_name: "gemini-2.5-flash",
          litellm_params: {
            model: "vertex_ai/gemini-2.5-flash",
          },
          model_info: {
            input_cost_per_token: 0.0,
            output_cost_per_token: 0.0,
            max_tokens: 65535,
            max_input_tokens: 1048576,
          },
        },
      ],
    };

    const result = transformModelData(rawData, mockGetProviderFromModel);

    // Zero costs should be converted to "0.00" per 1M tokens, not left as 0 or null
    expect(result.data[0]).toHaveProperty("input_cost", "0.00");
    expect(result.data[0]).toHaveProperty("output_cost", "0.00");
  });

  it("should handle null cost fields in model_info", () => {
    const rawData = {
      data: [
        {
          model_name: "some-model",
          litellm_params: {
            model: "openai/some-model",
          },
          model_info: {
            input_cost_per_token: null,
            output_cost_per_token: null,
            max_tokens: 4096,
            max_input_tokens: 8192,
          },
        },
      ],
    };

    const result = transformModelData(rawData, mockGetProviderFromModel);

    // Null costs should remain null (displayed as "-" in the UI)
    expect(result.data[0].input_cost).toBeNull();
    expect(result.data[0].output_cost).toBeNull();
  });

  it("keeps per-second pricing and resolution tiers for video models priced per second", () => {
    const rawData = {
      data: [
        {
          model_name: "veo-3.1-fast",
          litellm_params: { model: "vertex_ai/veo-3.1-fast-generate-001" },
          model_info: {
            input_cost_per_token: 0,
            output_cost_per_token: 0,
            output_cost_per_second: 0.1,
            output_cost_per_second_1080p: 0.12,
            output_cost_per_second_4k: 0.3,
          },
        },
        {
          model_name: "gpt-4",
          litellm_params: { model: "gpt-4" },
          model_info: { input_cost_per_token: 0.0000015, output_cost_per_token: 0.000002 },
        },
      ],
    };

    const result = transformModelData(rawData, mockGetProviderFromModel);

    expect(result.data[0].output_cost_per_second).toBe(0.1);
    expect(result.data[0].output_cost_per_second_tiers).toEqual([
      { resolution: "1080p", cost: 0.12 },
      { resolution: "4k", cost: 0.3 },
    ]);
    expect(result.data[1].output_cost_per_second).toBeNull();
    expect(result.data[1].output_cost_per_second_tiers).toEqual([]);
  });

  it("prefers a per-second override from litellm_params over model_info", () => {
    const rawData = {
      data: [
        {
          model_name: "veo-3.1",
          litellm_params: { model: "vertex_ai/veo-3.1-generate-001", output_cost_per_second: 0.5 },
          model_info: { output_cost_per_second: 0.4 },
        },
      ],
    };

    expect(transformModelData(rawData, mockGetProviderFromModel).data[0].output_cost_per_second).toBe(0.5);
  });

  it("should handle missing model_info", () => {
    const rawData = {
      data: [
        {
          model_name: "some-model",
          litellm_params: {
            model: "openai/some-model",
          },
        },
      ],
    };

    const result = transformModelData(rawData, mockGetProviderFromModel);

    // Missing model_info should result in null costs
    expect(result.data[0].input_cost).toBeNull();
    expect(result.data[0].output_cost).toBeNull();
  });
});
