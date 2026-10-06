import { z } from "zod";

export type DecisionEndpoint = "/v1/decisions" | "/typesafe/v1/systemone";

const decisionsJson = z.union([z.string(), z.record(z.string(), z.unknown()), z.array(z.unknown())]);
const decisionInstructions = decisionsJson.nullish();
const decisionQuestionSchema = z.discriminatedUnion("type", [
  z.looseObject({
    type: z.literal("choice"),
    instructions: decisionInstructions,
    criteria: z.record(z.string(), decisionsJson.nullable()).refine((value) => {
      const count = Object.keys(value).length;
      return count >= 1 && count <= 255;
    }, "Choice criteria must contain between 1 and 255 options."),
  }),
  z
    .looseObject({
      type: z.literal("noul"),
      instructions: decisionInstructions,
      criteria: z.partialRecord(z.enum(["true", "false"]), decisionsJson.nullable()).nullish(),
    })
    .refine((value) => value.instructions != null || value.criteria != null, {
      message: "A noul question requires instructions or criteria.",
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
    .refine((value) => value.trim().length > 0, "Model must be a non-empty string.")
    .optional(),
  state: decisionsJson,
  questions: z.record(z.string().min(1), decisionQuestionSchema).refine((value) => {
    const count = Object.keys(value).length;
    return count >= 1 && count <= 128;
  }, "Questions must contain between 1 and 128 entries."),
});

export type DecisionRequest = z.infer<typeof decisionsRequestSchema>;
export type PlaygroundRequest = SystemOneRequest | DecisionRequest;
export type PlaygroundQuestion = SystemOneQuestion | z.infer<typeof decisionQuestionSchema>;

const MAX_CHOICE_OPTIONS = 255;

const nonEmptyString = (message: string) =>
  z.string({ error: message }).refine((value) => value.trim().length > 0, message);

const instructions = z.string({
  error: (iss) =>
    iss.input === undefined ? "Required property 'instructions' is missing." : "Instructions must be a string.",
});

const choiceQuestionSchema = z.looseObject({
  type: z.literal("choice"),
  instructions,
  criteria: z
    .record(z.string(), z.string({ error: "Choice descriptions must be strings." }), {
      error: "Choice criteria must be an object mapping labels to descriptions.",
    })
    .refine((criteria) => {
      const count = Object.keys(criteria).length;
      return count >= 1 && count <= MAX_CHOICE_OPTIONS;
    }, `Choice criteria must contain between 1 and ${MAX_CHOICE_OPTIONS} options.`),
});

const noulQuestionSchema = z.looseObject({
  type: z.literal("noul"),
  instructions,
  criteria: z
    .looseObject(
      {
        true: z.string({ error: "Noul true criteria must be a string." }).optional(),
        false: z.string({ error: "Noul false criteria must be a string." }).optional(),
      },
      { error: "Noul criteria, when provided, must be an object." },
    )
    .optional(),
});

const scoreQuestionSchema = z.looseObject({
  type: z.literal("score"),
  instructions,
  criteria: z
    .array(z.string({ error: "Score levels must be strings." }), {
      error: "Score criteria must be an array of levels.",
    })
    .min(2, "Score criteria must contain at least 2 levels."),
});

const QUESTION_ERROR_MESSAGES: Partial<Record<z.core.$ZodIssue["code"], string>> = {
  invalid_union: "Question type must be choice, noul, or score.",
  invalid_type: "Question must be an object.",
};

const questionSchema = z.discriminatedUnion("type", [choiceQuestionSchema, noulQuestionSchema, scoreQuestionSchema], {
  error: (iss) => QUESTION_ERROR_MESSAGES[iss.code],
});

export const systemOneRequestSchema = z.looseObject(
  {
    model: nonEmptyString("Model must be a non-empty string.").optional(),
    state: z.unknown().nonoptional({ error: "Required property 'state' is missing." }),
    questions: z
      .record(z.string(), questionSchema, {
        error: (iss) =>
          iss.input === undefined
            ? "Required property 'questions' is missing."
            : "Questions must be a non-empty object.",
      })
      .refine((questions) => Object.keys(questions).length > 0, "At least one question is required."),
  },
  { error: "Payload must be a JSON object." },
);

const probability = z.number().finite().min(0).max(1);
const probabilities = z.record(z.string(), probability);

const noulAnswerSchema = z.looseObject({ type: z.literal("noul"), noul: probability });

const choiceAnswerShape = {
  type: z.literal("choice"),
  choice: z.string(),
  confidence: probability.optional(),
  probabilities,
};
const choiceAnswerSchema = z.looseObject(choiceAnswerShape);

const scoreAnswerShape = {
  type: z.literal("score"),
  score: z.number().finite(),
  confidence: probability.optional(),
  legend: z.record(z.string(), decisionsJson).optional(),
  probabilities,
};
const scoreAnswerSchema = z.looseObject(scoreAnswerShape);

const tokenCount = z.number().finite().nonnegative();

export const systemOneResponseSchema = z.looseObject({
  model: nonEmptyString("Model must be a non-empty string.").nullish(),
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
