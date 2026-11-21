import { createSerializer, parseAsArrayOf, parseAsBoolean, parseAsStringLiteral, type inferParserType } from "nuqs";
import { buildRequest, FOCUS, type RecommenderState } from "./api";

export const GOALS = ["quality", "fastest", "cheapest", "balanced"] as const;
export const PRESETS = {
  quality: { intelligence: 3, speed: 1, cost: 1 },
  fastest: { intelligence: 0, speed: 3, cost: 0 },
  cheapest: { intelligence: 1, speed: 0, cost: 3 },
  balanced: { intelligence: 2, speed: 2, cost: 2 },
} as const;
export const flowParsers = {
  step: parseAsStringLiteral(["1", "2", "3", "results"] as const).withDefault("1"),
  focus: parseAsArrayOf(parseAsStringLiteral(FOCUS)).withDefault(["general"]),
  goal: parseAsStringLiteral(GOALS).withDefault("balanced"),
  vision: parseAsBoolean.withDefault(false),
  function_calling: parseAsBoolean.withDefault(false),
  reasoning: parseAsBoolean.withDefault(false),
  long_context: parseAsBoolean.withDefault(false),
  include_external: parseAsBoolean.withDefault(false),
};
export type Answers = inferParserType<typeof flowParsers>;
export const EMPTY_ANSWERS: Answers = {
  step: "1",
  focus: ["general"],
  goal: "balanced",
  vision: false,
  function_calling: false,
  reasoning: false,
  long_context: false,
  include_external: false,
};
export const REQUIREMENTS = ["vision", "function_calling", "reasoning", "long_context"] as const;

export function answersToRequest(answers: Answers) {
  const state: RecommenderState = {
    ...PRESETS[answers.goal],
    focus: answers.focus.length ? answers.focus : ["general"],
    reasoning: answers.reasoning,
    vision: answers.vision,
    function_calling: answers.function_calling,
    context: answers.long_context ? 128000 : null,
    include_external: answers.include_external,
  };
  return buildRequest(state);
}

const serialize = createSerializer(flowParsers);
export function recommenderHref(focus?: (typeof FOCUS)[number]): string {
  return serialize("/model-recommender", focus ? { focus: [focus], step: "2" } : {});
}
