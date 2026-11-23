import { describe, expect, it } from "vitest";
import {
  answersToRequest,
  EMPTY_ANSWERS,
  flowParsers,
  recommenderHref,
  serializeAnswers,
  toggleValue,
  weightMix,
  type Answers,
} from "./flow";

describe("five-step recommendation answers", () => {
  it("maps custom priorities, providers and requirements without dropping zero", () => {
    const answers: Answers = {
      ...EMPTY_ANSWERS,
      intelligence: 0,
      speed: 3,
      cost: 1,
      focus: ["coding", "agentic"],
      vision: true,
      function_calling: true,
      reasoning: true,
      long_context: true,
      include_external: true,
      providers: ["openai"],
    };
    const expected = {
      priorities: { intelligence: 0, speed: 3, cost: 1 },
      focus: ["coding", "agentic"],
      requirements: { reasoning: true, vision: true, function_calling: true, min_context_tokens: 128000 },
      include_external: true,
      providers: ["openai"],
      limit: 10,
    };
    expect(answersToRequest(answers)).toEqual(expected);
  });
  it("keeps default requests gateway-only without a provider restriction", () => {
    const expected = {
      priorities: { intelligence: 2, speed: 2, cost: 2 },
      focus: ["general"],
      requirements: { reasoning: false, vision: false, function_calling: false, min_context_tokens: null },
      include_external: false,
      providers: [],
      limit: 10,
    };
    expect(answersToRequest({ ...EMPTY_ANSWERS, focus: [] })).toEqual(expected);
  });
  it("validates shared URLs and accepts old results links", () => {
    expect(flowParsers.step.parse("6")).toBeNull();
    expect(flowParsers.step.parse("results")).toBe("5");
    expect(flowParsers.intelligence.parse("-1")).toBeNull();
    expect(flowParsers.speed.parse("4")).toBeNull();
    expect(flowParsers.cost.parse("0")).toBe(0);
    expect(flowParsers.providers.parse("openai,anthropic")).toEqual(["openai", "anthropic"]);
    expect(flowParsers.focus.parse("coding,agentic")).toEqual(["coding", "agentic"]);
    expect(flowParsers.include_external.parse("true")).toBe(true);
    expect(recommenderHref("coding")).toBe("/model-recommender?focus=coding");
    const shared: Partial<Answers> = {
      step: "5",
      intelligence: 3,
      speed: 0,
      providers: ["openai"],
      include_external: true,
    };
    expect(serializeAnswers("/model-recommender", shared)).toBe(
      "/model-recommender?step=5&intelligence=3&speed=0&include_external=true&providers=openai",
    );
    expect(flowParsers.providers.parse(flowParsers.providers.serialize(["openai", "anthropic"]))).toEqual([
      "openai",
      "anthropic",
    ]);
  });
  it("shows proportional weights and a neutral empty mix without division by zero", () => {
    expect(weightMix({ intelligence: 3, speed: 1, cost: 0 })).toEqual([
      { key: "intelligence", percent: 75 },
      { key: "speed", percent: 25 },
      { key: "cost", percent: 0 },
    ]);
    expect(weightMix({ intelligence: 0, speed: 0, cost: 0 })).toEqual([
      { key: "intelligence", percent: 0 },
      { key: "speed", percent: 0 },
      { key: "cost", percent: 0 },
    ]);
    expect(toggleValue(["general", "coding"], "general")).toEqual(["coding"]);
    expect(toggleValue([], "coding")).toEqual(["coding"]);
  });
});
