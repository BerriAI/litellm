import { describe, expect, it } from "vitest";

import { estimateFields, estimateRules, withNormalizedEstimates } from "./estimatedOutputTokens";

const expectRejects = async (value: unknown) =>
  expect(estimateRules.perModel.validator(null, value)).rejects.toThrow(/JSON object of positive integers/);

describe("estimateFields", () => {
  it("renders a stored per-model map as editable JSON text", () => {
    expect(
      estimateFields({
        default_estimated_output_tokens: 2048,
        default_estimated_output_tokens_per_model: { "gpt-4": 4096 },
      }),
    ).toEqual({
      default_estimated_output_tokens: 2048,
      default_estimated_output_tokens_per_model: '{"gpt-4":4096}',
    });
  });

  it("leaves the controls blank when metadata carries neither setting", () => {
    expect(estimateFields({ unrelated: true })).toEqual({
      default_estimated_output_tokens: undefined,
      default_estimated_output_tokens_per_model: "",
    });
  });

  it("tolerates absent metadata", () => {
    expect(estimateFields(null).default_estimated_output_tokens_per_model).toBe("");
    expect(estimateFields(undefined).default_estimated_output_tokens_per_model).toBe("");
  });
});

describe("estimateRules.perModel", () => {
  it("accepts a blank control", async () => {
    await expect(estimateRules.perModel.validator(null, "")).resolves.toBeUndefined();
    await expect(estimateRules.perModel.validator(null, "   ")).resolves.toBeUndefined();
    await expect(estimateRules.perModel.validator(null, undefined)).resolves.toBeUndefined();
  });

  it("accepts a per-model object", async () => {
    await expect(estimateRules.perModel.validator(null, '{"gpt-4": 4096}')).resolves.toBeUndefined();
  });

  it("rejects text that is not JSON", async () => {
    await expectRejects("gpt-4: 4096");
  });

  it("rejects JSON that is not an object, which the API would refuse", async () => {
    await expectRejects("4096");
    await expectRejects('"gpt-4"');
    await expectRejects("[4096]");
    await expectRejects("null");
  });

  it("rejects a per-model map whose values the runtime would ignore", async () => {
    await expectRejects('{"gpt-4": -5}');
    await expectRejects('{"gpt-4": 0}');
    await expectRejects('{"gpt-4": 4.5}');
    await expectRejects('{"gpt-4": "4096"}');
    await expectRejects("{}");
  });
});

describe("withNormalizedEstimates", () => {
  it("coerces the numeric control and parses the per-model control without mutating the input", () => {
    const values = {
      default_estimated_output_tokens: "2048",
      default_estimated_output_tokens_per_model: '{"gpt-4": 4096}',
      other: "untouched",
    };
    const before = { ...values };

    expect(withNormalizedEstimates(values)).toEqual({
      default_estimated_output_tokens: 2048,
      default_estimated_output_tokens_per_model: { "gpt-4": 4096 },
      other: "untouched",
    });
    expect(values).toEqual(before);
  });

  it("drops blank controls so a save never sends an empty value", () => {
    expect(
      withNormalizedEstimates({
        default_estimated_output_tokens: "",
        default_estimated_output_tokens_per_model: "   ",
      }),
    ).toEqual({});
  });

  it("drops each control independently", () => {
    expect(
      withNormalizedEstimates({
        default_estimated_output_tokens: 900,
        default_estimated_output_tokens_per_model: "",
      }),
    ).toEqual({ default_estimated_output_tokens: 900 });
  });

  it("drops a per-model map the API would reject rather than sending it", () => {
    expect(
      withNormalizedEstimates({
        default_estimated_output_tokens_per_model: '{"gpt-4": -5}',
      }),
    ).toEqual({});
  });
});

describe("estimateRules.positive", () => {
  it("accepts a blank control and a positive integer", async () => {
    await expect(estimateRules.positive.validator(null, "")).resolves.toBeUndefined();
    await expect(estimateRules.positive.validator(null, 2048)).resolves.toBeUndefined();
  });

  it("rejects values the runtime would ignore", async () => {
    await expect(estimateRules.positive.validator(null, 0)).rejects.toThrow(/positive integer/);
    await expect(estimateRules.positive.validator(null, -5)).rejects.toThrow(/positive integer/);
    await expect(estimateRules.positive.validator(null, 12.5)).rejects.toThrow(/positive integer/);
  });
});
