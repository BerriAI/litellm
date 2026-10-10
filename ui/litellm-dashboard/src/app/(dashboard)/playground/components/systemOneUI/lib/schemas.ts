import { z } from "zod";

export type DecisionEndpoint = "/v1/decisions" | "/v1/systemone" | "/typesafe/v1/systemone";

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

const openAIName = z.string().nullish();
const openAIInstructions = z.string({
  error: (iss) =>
    iss.input === undefined ? "Required property 'instructions' is missing." : "Instructions must be a string.",
});
const openAIChoiceValue = z.union([z.string(), z.boolean()]);
const openAIChoiceOptionSchema = z.looseObject({ value: openAIChoiceValue, description: z.string().nullish() });
const openAIScoreLevelSchema = z.looseObject({ label: z.string(), description: z.string().nullish() });
const openAIQuestionSchema = z.discriminatedUnion("type", [
  z.looseObject({ type: z.literal("predicate"), name: openAIName, instructions: openAIInstructions }),
  z.looseObject({
    type: z.literal("choice"),
    name: openAIName,
    instructions: openAIInstructions,
    choices: z
      .array(openAIChoiceOptionSchema, { error: "Choices must be an array of { value, description } options." })
      .min(2, "Choice questions need between 2 and 255 choices.")
      .max(255, "Choice questions need between 2 and 255 choices."),
  }),
  z.looseObject({
    type: z.literal("score"),
    name: openAIName,
    instructions: openAIInstructions,
    levels: z
      .array(openAIScoreLevelSchema, { error: "Levels must be an array of { label, description } levels." })
      .min(2, "Score questions need between 2 and 10 levels.")
      .max(10, "Score questions need between 2 and 10 levels."),
  }),
]);
const openAIInputPartSchema = z.discriminatedUnion("type", [
  z.looseObject({ type: z.literal("input_text"), text: z.string() }),
  z.looseObject({ type: z.literal("input_image"), image_url: z.string(), detail: z.string().nullish() }),
]);
const openAIInputMessageSchema = z.looseObject({
  role: z.literal("user").optional(),
  type: z.literal("message").optional(),
  content: z.union([z.string(), z.array(openAIInputPartSchema).min(1)]),
});

export const openAIDecisionsRequestSchema = z.looseObject({
  model: z
    .string()
    .refine((value) => value.trim().length > 0, "Model must be a non-empty string.")
    .optional(),
  input: z.union([z.string(), z.array(openAIInputMessageSchema).min(1)], {
    error: (iss) =>
      iss.input === undefined ? "Required property 'input' is missing." : "Input must be a string or a message list.",
  }),
  questions: z
    .array(openAIQuestionSchema, {
      error: (iss) =>
        iss.input === undefined ? "Required property 'questions' is missing." : "Questions must be an array.",
    })
    .min(1, "At least one question is required.")
    .max(128, "Questions must contain between 1 and 128 entries."),
  safety_identifier: z.string().nullish(),
});

export type OpenAIDecisionsRequest = z.infer<typeof openAIDecisionsRequestSchema>;
export type OpenAIDecisionQuestion = z.infer<typeof openAIQuestionSchema>;
export type PlaygroundRequest = SystemOneRequest | DecisionRequest | OpenAIDecisionsRequest;
export type PlaygroundQuestion = SystemOneQuestion | z.infer<typeof decisionQuestionSchema>;

export const isOpenAIDecisionsRequest = (payload: PlaygroundRequest): payload is OpenAIDecisionsRequest =>
  Array.isArray(payload.questions);

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

const openAIAnswerSchema = z.discriminatedUnion("type", [
  z.looseObject({ type: z.literal("predicate"), name: openAIName, probability }),
  z.looseObject({
    type: z.literal("choice"),
    name: openAIName,
    choice: openAIChoiceValue,
    confidence: probability.optional(),
    probabilities: z.array(z.looseObject({ value: openAIChoiceValue, probability })),
  }),
  z.looseObject({
    type: z.literal("score"),
    name: openAIName,
    score: z.number().finite(),
    confidence: probability.optional(),
    probabilities: z.array(z.looseObject({ value: z.number().finite(), label: z.string(), probability })),
  }),
  z.looseObject({ type: z.literal("refusal"), name: openAIName }),
]);

export const openAIDecisionsResponseSchema = z.looseObject({
  model: nonEmptyString("Model must be a non-empty string.").nullish(),
  answers: z.array(openAIAnswerSchema),
  usage: z.looseObject({ input_tokens: tokenCount, output_tokens: tokenCount }).nullish(),
});

export type OpenAIDecisionsResponse = z.infer<typeof openAIDecisionsResponseSchema>;
export type OpenAIDecisionAnswer = OpenAIDecisionsResponse["answers"][number];
export type PlaygroundResponse = SystemOneResponse | OpenAIDecisionsResponse;

export const isOpenAIDecisionsResponse = (response: PlaygroundResponse): response is OpenAIDecisionsResponse =>
  Array.isArray(response.answers);

export type SystemOneRequest = z.infer<typeof systemOneRequestSchema>;
export type SystemOneQuestion = z.infer<typeof questionSchema>;
export type SystemOneResponse = z.infer<typeof systemOneResponseSchema>;
export type SystemOneAnswer = SystemOneResponse["answers"][string];
