import { z } from "zod";

const contentPart = z.discriminatedUnion("type", [
  z.strictObject({ type: z.literal("input_text"), text: z.string() }),
  z.strictObject({ type: z.literal("input_image"), image_url: z.string(), detail: z.string().nullish() }),
]);
const inputMessage = z.strictObject({
  role: z.literal("user").optional(),
  type: z.literal("message").optional(),
  content: z.union([z.string(), z.array(contentPart)]),
});
const questionFields = { name: z.string().nullish(), instructions: z.string() };
const choiceValue = z.union([z.string(), z.boolean()]);
const choiceQuestion = z
  .strictObject({
    ...questionFields,
    type: z.literal("choice"),
    choices: z
      .array(z.strictObject({ value: choiceValue, description: z.string().nullish() }))
      .min(2)
      .max(255),
  })
  .refine(
    (question) => new Set(question.choices.map((choice) => String(choice.value))).size === question.choices.length,
    {
      message: "Choice values must be unique, including their string representations.",
      path: ["choices"],
    },
  );

const requestFields = {
  model: z
    .string()
    .refine((value) => value.trim().length > 0, "Model must be a non-empty string.")
    .optional(),
  input: z.union([z.string(), z.array(inputMessage)]),
  questions: z
    .array(
      z.discriminatedUnion("type", [
        z.strictObject({ ...questionFields, type: z.literal("predicate") }),
        choiceQuestion,
        z.strictObject({
          ...questionFields,
          type: z.literal("score"),
          levels: z
            .array(z.strictObject({ label: z.string(), description: z.string().nullish() }))
            .min(2)
            .max(10),
        }),
      ]),
    )
    .min(1)
    .max(128),
  safety_identifier: z.string().nullish(),
};
export const openAIDecisionsRequestSchema = z.looseObject(requestFields);

const probability = z.number().finite().min(0).max(1);
const answerFields = { name: z.string().nullable() };
const tokenCount = z.number().int().nonnegative();
const choiceAnswerFields = {
  ...answerFields,
  type: z.literal("choice"),
  choice: choiceValue,
  confidence: probability,
  probabilities: z.array(z.looseObject({ value: choiceValue, probability })),
};
const scoreAnswerFields = {
  ...answerFields,
  type: z.literal("score"),
  score: z.number().finite(),
  confidence: probability,
  probabilities: z.array(z.looseObject({ value: z.number().int(), label: z.string(), probability })),
};
const usageFields = {
  input_tokens: tokenCount,
  output_tokens: tokenCount,
  total_tokens: tokenCount,
  input_tokens_details: z
    .looseObject({ cached_tokens: tokenCount.optional(), cache_write_tokens: tokenCount.optional() })
    .optional(),
  output_tokens_details: z.looseObject({ reasoning_tokens: tokenCount.optional() }).optional(),
};
export const openAIDecisionsResponseSchema = z.looseObject({
  model: z.string(),
  answers: z.array(
    z.discriminatedUnion("type", [
      z.looseObject({ ...answerFields, type: z.literal("predicate"), probability }),
      z.looseObject(choiceAnswerFields),
      z.looseObject(scoreAnswerFields),
      z.looseObject({ ...answerFields, type: z.literal("refusal") }),
    ]),
  ),
  usage: z.looseObject(usageFields),
});

export type OpenAIDecisionsRequest = z.infer<typeof openAIDecisionsRequestSchema>;
export type OpenAIDecisionsResponse = z.infer<typeof openAIDecisionsResponseSchema>;
