import { describe, expect, it } from "vitest";
import { answersToRequest, EMPTY_ANSWERS, flowParsers, PRESETS, recommenderHref, type Answers } from "./flow";

describe("guided recommendation answers", () => {
  it("maps all priority presets and optional hard requirements", () => {
    for (const goal of ["quality", "fastest", "cheapest", "balanced"] as const) {
      const answers: Answers = {
        ...EMPTY_ANSWERS,
        goal,
        focus: ["coding", "agentic"],
        vision: true,
        function_calling: true,
        reasoning: true,
        long_context: true,
        include_external: true,
      };
      const request = answersToRequest(answers);
      expect(request.priorities).toEqual(PRESETS[goal]);
      expect(request.focus).toEqual(["coding", "agentic"]);
      const requirements = {
        reasoning: true,
        vision: true,
        function_calling: true,
        min_context_tokens: 128000,
      };
      expect(request.requirements).toEqual(requirements);
      expect(request.include_external).toBe(true);
    }
  });
  it("defaults to gateway-only balanced general chat without hard requirements", () => {
    const request = answersToRequest({ ...EMPTY_ANSWERS, focus: [] });
    expect(request.focus).toEqual(["general"]);
    expect(request.include_external).toBe(false);
    expect(request.requirements?.min_context_tokens).toBeNull();
  });
  it("rejects malformed URLs and restores valid shared results", () => {
    expect(flowParsers.step.parse("4")).toBeNull();
    expect(flowParsers.goal.parse("unknown")).toBeNull();
    expect(flowParsers.include_external.parse("maybe")).toBe(false);
    expect(flowParsers.focus.parse("coding,agentic")).toEqual(["coding", "agentic"]);
    expect(flowParsers.step.parse("results")).toBe("results");
    expect(flowParsers.include_external.parse("true")).toBe(true);
    expect(recommenderHref("coding")).toBe("/model-recommender?step=2&focus=coding");
  });
});
