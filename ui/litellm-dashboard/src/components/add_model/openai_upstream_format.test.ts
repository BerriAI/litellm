import { describe, expect, it } from "vitest";
import { OpenAIUpstreamFormat, parseOpenAIUpstreamModel, serializeOpenAIUpstreamModel } from "./openai_upstream_format";

describe("OpenAI upstream format", () => {
  it.each([
    "org/model",
    "openai/org/model",
    "openai/responses/org/model",
    "openai/systemone/org/model",
    "openai/decisions/org/model",
    "openai/systemone/systemone/model",
    "openai/systemone/decisions/model",
    "openai/decisions/decisions/model",
    "openai/decisions/systemone/model",
  ])("should preserve %s when an existing model is saved without changes", (model) => {
    const parsed = parseOpenAIUpstreamModel(model);
    expect(serializeOpenAIUpstreamModel(parsed.model, parsed.format)).toBe(model);
  });

  it.each([OpenAIUpstreamFormat.SystemOne, OpenAIUpstreamFormat.Decisions])(
    "should preserve the full upstream model name when selecting %s",
    (format) => {
      expect(serializeOpenAIUpstreamModel("org/team/model", format)).toBe(`openai/${format}/org/team/model`);
      expect(serializeOpenAIUpstreamModel("openai/org/team/model", format)).toBe(`openai/${format}/org/team/model`);
    },
  );

  it("should replace the selected format without stacking routing prefixes", () => {
    const parsed = parseOpenAIUpstreamModel("openai/systemone/org/model");
    expect(serializeOpenAIUpstreamModel(parsed.model, OpenAIUpstreamFormat.Decisions)).toBe(
      "openai/decisions/org/model",
    );
  });

  it("should restore the provider default while preserving the model name", () => {
    const parsed = parseOpenAIUpstreamModel("openai/decisions/org/model");
    expect(serializeOpenAIUpstreamModel(parsed.model, OpenAIUpstreamFormat.Default)).toBe("openai/org/model");
  });
});
