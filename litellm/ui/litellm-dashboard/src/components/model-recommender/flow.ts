import {
  createParser,
  createSerializer,
  parseAsArrayOf,
  parseAsBoolean,
  parseAsString,
  parseAsStringLiteral,
  type inferParserType,
} from "nuqs";
import { buildRequest, FOCUS, PRIORITIES, recommenderParsers, type RecommenderState } from "./api";

export const STEPS = ["1", "2", "3", "4", "5"] as const;
export const flowParsers = {
  step: createParser({
    parse: (value) => (value === "results" ? ("5" as const) : STEPS.find((step) => step === value) ?? null),
    serialize: String,
  }).withDefault("1"),
  intelligence: recommenderParsers.intelligence,
  speed: recommenderParsers.speed,
  cost: recommenderParsers.cost,
  focus: parseAsArrayOf(parseAsStringLiteral(FOCUS)).withDefault(["general"]),
  vision: parseAsBoolean.withDefault(false),
  function_calling: parseAsBoolean.withDefault(false),
  reasoning: parseAsBoolean.withDefault(false),
  long_context: parseAsBoolean.withDefault(false),
  include_external: parseAsBoolean.withDefault(false),
  providers: parseAsArrayOf(parseAsString).withDefault([]),
};
export type Answers = inferParserType<typeof flowParsers>;
export const EMPTY_ANSWERS: Answers = {
  step: "1",
  focus: ["general"],
  intelligence: 2,
  speed: 2,
  cost: 2,
  vision: false,
  function_calling: false,
  reasoning: false,
  long_context: false,
  include_external: false,
  providers: [],
};
export const REQUIREMENTS = ["reasoning", "vision", "function_calling", "long_context"] as const;

export function answersToRequest(answers: Answers) {
  const state: RecommenderState = {
    ...answers,
    focus: answers.focus.length ? answers.focus : ["general"],
    reasoning: answers.reasoning,
    vision: answers.vision,
    function_calling: answers.function_calling,
    context: answers.long_context ? 128000 : null,
    include_external: answers.include_external,
  };
  return buildRequest(state);
}

export function weightMix(priorities: Pick<Answers, (typeof PRIORITIES)[number]>) {
  const total = PRIORITIES.reduce((sum, key) => sum + priorities[key], 0);
  return PRIORITIES.map((key) => ({ key, percent: total ? (priorities[key] / total) * 100 : 0 }));
}

export function toggleValue<T extends string>(values: readonly T[], value: T): T[] {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

export const serializeAnswers = createSerializer(flowParsers);
export function recommenderHref(focus?: (typeof FOCUS)[number]): string {
  return serializeAnswers("/model-recommender", focus ? { focus: [focus] } : {});
}
