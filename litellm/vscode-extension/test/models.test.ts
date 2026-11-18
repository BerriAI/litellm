import { describe, expect, it } from "vitest";
import {
  ASSUMED_MAX_INPUT_TOKENS,
  ASSUMED_MAX_OUTPUT_TOKENS,
  MARKDOWN_LINE_BREAK,
  describeModels,
  estimateTokens,
  formatUsdPerMillionTokens,
  parseModelGroups,
  reasoningEffortFrom,
  type ModelGroupInfo,
} from "../src/models";

const gatewayGroup = (overrides: Partial<Record<string, unknown>> = {}): Record<string, unknown> => ({
  model_group: "gpt-5.6",
  providers: ["openai"],
  max_input_tokens: 922000,
  max_output_tokens: 128000,
  input_cost_per_token: 4e-6,
  output_cost_per_token: 2e-5,
  mode: "chat",
  supports_vision: true,
  supports_function_calling: true,
  supports_reasoning: true,
  supported_reasoning_efforts: ["none", "low", "medium", "high", "xhigh"],
  ...overrides,
});

const parsed = (...groups: readonly Record<string, unknown>[]): readonly ModelGroupInfo[] => {
  const result = parseModelGroups({ data: groups });
  if (result.kind !== "ok") {
    throw new Error(result.reason);
  }
  return result.groups;
};

describe("parseModelGroups", () => {
  it("maps the gateway's /model_group/info shape", () => {
    expect(parsed(gatewayGroup())).toEqual([
      {
        modelGroup: "gpt-5.6",
        providers: ["openai"],
        mode: "chat",
        maxInputTokens: 922000,
        maxOutputTokens: 128000,
        inputCostPerToken: 4e-6,
        outputCostPerToken: 2e-5,
        supportsVision: true,
        supportsFunctionCalling: true,
        supportedReasoningEfforts: ["none", "low", "medium", "high", "xhigh"],
      },
    ]);
  });

  it("treats null limits, prices, and efforts as unknown", () => {
    const [group] = parsed(
      gatewayGroup({
        max_input_tokens: null,
        max_output_tokens: null,
        input_cost_per_token: null,
        output_cost_per_token: null,
        supported_reasoning_efforts: null,
        supports_vision: null,
      }),
    );
    expect(group).toMatchObject({
      maxInputTokens: undefined,
      inputCostPerToken: undefined,
      supportedReasoningEfforts: [],
      supportsVision: false,
    });
  });

  it("drops entries without a model_group and rejects payloads without data", () => {
    expect(parsed({ providers: ["openai"] }, gatewayGroup()).map((group) => group.modelGroup)).toEqual(["gpt-5.6"]);
    expect(parseModelGroups({ detail: "Unauthorized" })).toEqual({ kind: "invalid", reason: "response has no data array" });
  });
});

describe("describeModels", () => {
  it("lists chat groups with USD pricing in the detail and a full tooltip", () => {
    const [model] = describeModels(parsed(gatewayGroup()));
    expect(model).toMatchObject({
      id: "gpt-5.6",
      name: "gpt-5.6",
      family: "gpt-5.6",
      detail: "$4.00 in / $20.00 out per 1M tokens",
      maxInputTokens: 922000,
      maxOutputTokens: 128000,
      imageInput: true,
      toolCalling: true,
    });
    expect(model?.tooltip).toBe(
      [
        "LiteLLM model group gpt-5.6 via openai",
        "Input: $4.00 per 1M tokens",
        "Output: $20.00 per 1M tokens",
        "Context: 922000 in / 128000 out tokens",
        "Reasoning effort: none, low, medium, high, xhigh",
      ].join(MARKDOWN_LINE_BREAK),
    );
  });

  it("offers the gateway's reasoning efforts behind a gateway default entry", () => {
    const [model] = describeModels(parsed(gatewayGroup({ supported_reasoning_efforts: ["low", "high"] })));
    expect(model?.configurationSchema).toEqual({
      properties: {
        reasoningEffort: {
          type: "string",
          title: "Reasoning Effort",
          enum: ["default", "low", "high"],
          enumItemLabels: ["Gateway default", "Low", "High"],
          default: "default",
          group: "navigation",
        },
      },
    });
  });

  it("has no configuration schema when the group lists no reasoning efforts", () => {
    const [model] = describeModels(parsed(gatewayGroup({ supported_reasoning_efforts: null })));
    expect(model?.configurationSchema).toBeUndefined();
    expect(model?.tooltip).toContain("Reasoning effort: not configurable");
  });

  it("keeps groups without a mode and skips non-chat groups", () => {
    const models = describeModels(
      parsed(
        gatewayGroup({ model_group: "text-embedding-4", mode: "embedding" }),
        gatewayGroup({ model_group: "whisper-3", mode: "audio_transcription" }),
        gatewayGroup({ model_group: "gpt-image-2", mode: "image_generation" }),
        gatewayGroup({ model_group: "unlabeled", mode: null }),
        gatewayGroup(),
      ),
    );
    expect(models.map((model) => model.id)).toEqual(["unlabeled", "gpt-5.6"]);
  });

  it("falls back to assumed context limits and says so", () => {
    const [model] = describeModels(parsed(gatewayGroup({ max_input_tokens: null, max_output_tokens: null })));
    expect(model).toMatchObject({ maxInputTokens: ASSUMED_MAX_INPUT_TOKENS, maxOutputTokens: ASSUMED_MAX_OUTPUT_TOKENS });
    expect(model?.tooltip).toContain(`Context: unknown, assuming ${ASSUMED_MAX_INPUT_TOKENS} in / ${ASSUMED_MAX_OUTPUT_TOKENS} out tokens`);
  });

  it("shows missing prices instead of inventing zeros", () => {
    const [both, inputOnly, free] = describeModels(
      parsed(
        gatewayGroup({ input_cost_per_token: null, output_cost_per_token: null }),
        gatewayGroup({ output_cost_per_token: null }),
        gatewayGroup({ input_cost_per_token: 0, output_cost_per_token: 0 }),
      ),
    );
    expect(both?.detail).toBe("No pricing configured");
    expect(both?.tooltip).toContain("Input: no price configured");
    expect(inputOnly?.detail).toBe("$4.00 in / n/a out per 1M tokens");
    expect(free?.detail).toBe("$0.00 in / $0.00 out per 1M tokens");
  });
});

describe("formatUsdPerMillionTokens", () => {
  it("renders cents for ordinary prices and two significant digits below a cent", () => {
    expect(formatUsdPerMillionTokens(4e-6)).toBe("$4.00");
    expect(formatUsdPerMillionTokens(7.5e-7)).toBe("$0.75");
    expect(formatUsdPerMillionTokens(2.5e-5)).toBe("$25.00");
    expect(formatUsdPerMillionTokens(1e-9)).toBe("$0.0010");
    expect(formatUsdPerMillionTokens(0)).toBe("$0.00");
  });
});

describe("reasoningEffortFrom", () => {
  it("forwards a chosen effort and leaves the gateway default unset", () => {
    expect(reasoningEffortFrom({ reasoningEffort: "high" })).toBe("high");
    expect(reasoningEffortFrom({ reasoningEffort: "default" })).toBeUndefined();
    expect(reasoningEffortFrom({ reasoningEffort: 3 })).toBeUndefined();
    expect(reasoningEffortFrom(undefined)).toBeUndefined();
  });
});

describe("estimateTokens", () => {
  it("rounds four characters per token upward", () => {
    expect(estimateTokens("")).toBe(0);
    expect(estimateTokens("abcd")).toBe(1);
    expect(estimateTokens("abcde")).toBe(2);
  });
});
