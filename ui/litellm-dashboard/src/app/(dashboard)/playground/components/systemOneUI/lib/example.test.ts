import { describe, expect, it } from "vitest";
import {
  DECISION_PRESETS,
  emptyPayload,
  openAIDecisionsExample,
  presetPayload,
  toOpenAIDecisionsRequest,
} from "./example";
import { validateOpenAIDecisionsPayload, validateSystemOnePayload } from "./validatePayload";

describe("decision presets", () => {
  it.each(DECISION_PRESETS.map((preset) => [preset.label, preset] as const))(
    "%s is a valid request on both endpoints",
    (_label, preset) => {
      expect(
        validateSystemOnePayload(presetPayload(preset, "/v1/systemone", "jev-latest"), "/v1/systemone"),
      ).toMatchObject({ isValid: true, issues: [] });
      expect(validateOpenAIDecisionsPayload(presetPayload(preset, "/v1/decisions", "jev-latest"))).toMatchObject({
        isValid: true,
        issues: [],
      });
    },
  );

  it("covers every question type across the presets", () => {
    const types = new Set(
      DECISION_PRESETS.flatMap((preset) => Object.values(preset.request.questions).map((question) => question.type)),
    );
    expect([...types].sort()).toEqual(["choice", "noul", "score"]);
  });

  it("maps System One questions onto the OpenAI shape by name", () => {
    const request = toOpenAIDecisionsRequest(DECISION_PRESETS[0].request, "jev-latest");
    expect(request.questions.map((question) => [question.type, question.name])).toEqual([
      ["choice", "area"],
      ["predicate", "has_repro_steps"],
      ["score", "severity"],
    ]);
    expect(request).toEqual(openAIDecisionsExample("jev-latest"));
  });

  it("omits the model from payloads when none is known", () => {
    expect(JSON.parse(presetPayload(DECISION_PRESETS[1], "/v1/systemone"))).not.toHaveProperty("model");
    expect(JSON.parse(emptyPayload("/v1/decisions"))).toEqual({ input: "", questions: [] });
    expect(validateSystemOnePayload(emptyPayload("/v1/systemone"), "/v1/systemone").isValid).toBe(false);
  });
});
