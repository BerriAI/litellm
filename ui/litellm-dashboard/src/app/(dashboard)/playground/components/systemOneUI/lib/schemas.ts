import type { ParseKeys } from "i18next";
import { z } from "zod";
import i18n from "@/i18n/i18n";

export type DecisionEndpoint = "/v1/systemone" | "/typesafe/v1/systemone";

const decisionsJson = z.union([z.string(), z.record(z.string(), z.unknown()), z.array(z.unknown())]);
const decisionInstructions = decisionsJson.nullish();
const decisionQuestionSchema = z.discriminatedUnion("type", [
  z.looseObject({
    type: z.literal("choice"),
    instructions: decisionInstructions,
    criteria: z.record(z.string(), decisionsJson.nullable()).refine(
      (value) => {
        const count = Object.keys(value).length;
        return count >= 1 && count <= 255;
      },
      { error: () => i18n.t("playground.systemOne.schemas.choiceOptionCount", { max: 255 }) },
    ),
  }),
  z
    .looseObject({
      type: z.literal("noul"),
      instructions: decisionInstructions,
      criteria: z.partialRecord(z.enum(["true", "false"]), decisionsJson.nullable()).nullish(),
    })
    .refine((value) => value.instructions != null || value.criteria != null, {
      error: () => i18n.t("playground.systemOne.schemas.noulNeedsInstructionsOrCriteria"),
    }),
  z.looseObject({
    type: z.literal("score"),
    instructions: decisionInstructions,
    criteria: z.array(decisionsJson).min(1).max(10),
  }),
]);

export const decisionsRequestSchema = z.looseObject({
  model: z
    .string()
    .refine((value) => value.trim().length > 0, { error: () => i18n.t("playground.systemOne.schemas.modelNonEmpty") })
    .optional(),
  state: decisionsJson,
  questions: z.record(z.string().min(1), decisionQuestionSchema).refine(
    (value) => {
      const count = Object.keys(value).length;
      return count >= 1 && count <= 128;
    },
    { error: () => i18n.t("playground.systemOne.schemas.questionCount") },
  ),
});

export type DecisionRequest = z.infer<typeof decisionsRequestSchema>;
export type PlaygroundRequest = SystemOneRequest | DecisionRequest;
export type PlaygroundQuestion = SystemOneQuestion | z.infer<typeof decisionQuestionSchema>;

const MAX_CHOICE_OPTIONS = 255;

const nonEmptyString = (messageKey: ParseKeys) =>
  z
    .string({ error: () => i18n.t(messageKey) })
    .refine((value) => value.trim().length > 0, { error: () => i18n.t(messageKey) });

const instructions = z.string({
  error: (iss) =>
    iss.input === undefined
      ? i18n.t("playground.systemOne.schemas.instructionsMissing")
      : i18n.t("playground.systemOne.schemas.instructionsType"),
});

const choiceQuestionSchema = z.looseObject({
  type: z.literal("choice"),
  instructions,
  criteria: z
    .record(z.string(), z.string({ error: () => i18n.t("playground.systemOne.schemas.choiceDescriptionType") }), {
      error: () => i18n.t("playground.systemOne.schemas.choiceCriteriaType"),
    })
    .refine(
      (criteria) => {
        const count = Object.keys(criteria).length;
        return count >= 1 && count <= MAX_CHOICE_OPTIONS;
      },
      { error: () => i18n.t("playground.systemOne.schemas.choiceOptionCount", { max: MAX_CHOICE_OPTIONS }) },
    ),
});

const noulQuestionSchema = z.looseObject({
  type: z.literal("noul"),
  instructions,
  criteria: z
    .looseObject(
      {
        true: z.string({ error: () => i18n.t("playground.systemOne.schemas.noulTrueType") }).optional(),
        false: z.string({ error: () => i18n.t("playground.systemOne.schemas.noulFalseType") }).optional(),
      },
      { error: () => i18n.t("playground.systemOne.schemas.noulCriteriaType") },
    )
    .optional(),
});

const scoreQuestionSchema = z.looseObject({
  type: z.literal("score"),
  instructions,
  criteria: z
    .array(z.string({ error: () => i18n.t("playground.systemOne.schemas.scoreLevelType") }), {
      error: () => i18n.t("playground.systemOne.schemas.scoreCriteriaType"),
    })
    .min(2, { error: () => i18n.t("playground.systemOne.schemas.scoreLevelCount") }),
});

const QUESTION_ERROR_MESSAGES: Partial<Record<z.core.$ZodIssue["code"], ParseKeys>> = {
  invalid_union: "playground.systemOne.schemas.questionType",
  invalid_type: "playground.systemOne.schemas.questionObject",
};

const questionSchema = z.discriminatedUnion("type", [choiceQuestionSchema, noulQuestionSchema, scoreQuestionSchema], {
  error: (iss) => {
    const key = QUESTION_ERROR_MESSAGES[iss.code];
    return key === undefined ? undefined : i18n.t(key);
  },
});

export const systemOneRequestSchema = z.looseObject(
  {
    model: nonEmptyString("playground.systemOne.schemas.modelNonEmpty").optional(),
    state: z.unknown().nonoptional({ error: () => i18n.t("playground.systemOne.schemas.stateMissing") }),
    questions: z
      .record(z.string(), questionSchema, {
        error: (iss) =>
          iss.input === undefined
            ? i18n.t("playground.systemOne.schemas.questionsMissing")
            : i18n.t("playground.systemOne.schemas.questionsType"),
      })
      .refine((questions) => Object.keys(questions).length > 0, {
        error: () => i18n.t("playground.systemOne.schemas.questionRequired"),
      }),
  },
  { error: () => i18n.t("playground.systemOne.schemas.payloadType") },
);

const probability = z.number().finite().min(0).max(1);
const probabilities = z.record(z.string(), probability);

const noulAnswerSchema = z.looseObject({ type: z.literal("noul"), noul: probability });

const choiceAnswerShape = {
  type: z.literal("choice"),
  choice: z.string(),
  confidence: probability.nullish(),
  probabilities,
};
const choiceAnswerSchema = z.looseObject(choiceAnswerShape);

const scoreAnswerShape = {
  type: z.literal("score"),
  score: z.number().finite(),
  confidence: probability.nullish(),
  legend: z.record(z.string(), decisionsJson).optional(),
  probabilities,
};
const scoreAnswerSchema = z.looseObject(scoreAnswerShape);

const tokenCount = z.number().finite().nonnegative();

export const systemOneResponseSchema = z.looseObject({
  model: nonEmptyString("playground.systemOne.schemas.modelNonEmpty").nullish(),
  answers: z.record(
    z.string(),
    z.discriminatedUnion("type", [noulAnswerSchema, choiceAnswerSchema, scoreAnswerSchema]),
  ),
  usage: z.looseObject({ input_tokens: tokenCount, output_tokens: tokenCount }).nullish(),
});

export type SystemOneRequest = z.infer<typeof systemOneRequestSchema>;
export type SystemOneQuestion = z.infer<typeof questionSchema>;
export type SystemOneResponse = z.infer<typeof systemOneResponseSchema>;
export type SystemOneAnswer = SystemOneResponse["answers"][string];
