import { z } from "zod";

const MAX_CHOICE_OPTIONS = 255;

const nonEmptyString = (message: string) =>
  z.string({ invalid_type_error: message }).refine((value) => value.trim().length > 0, message);

const instructions = z.string({
  required_error: "Required property 'instructions' is missing.",
  invalid_type_error: "Instructions must be a string.",
});

const choiceQuestionSchema = z
  .object({
    type: z.literal("choice"),
    instructions,
    criteria: z
      .record(z.string({ invalid_type_error: "Choice descriptions must be strings." }), {
        required_error: "Choice criteria must be an object mapping labels to descriptions.",
        invalid_type_error: "Choice criteria must be an object mapping labels to descriptions.",
      })
      .refine((criteria) => {
        const count = Object.keys(criteria).length;
        return count >= 1 && count <= MAX_CHOICE_OPTIONS;
      }, `Choice criteria must contain between 1 and ${MAX_CHOICE_OPTIONS} options.`),
  })
  .passthrough();

const noulQuestionSchema = z
  .object({
    type: z.literal("noul"),
    instructions,
    criteria: z
      .object(
        {
          true: z.string({ invalid_type_error: "Noul true criteria must be a string." }).optional(),
          false: z.string({ invalid_type_error: "Noul false criteria must be a string." }).optional(),
        },
        { invalid_type_error: "Noul criteria, when provided, must be an object." },
      )
      .passthrough()
      .optional(),
  })
  .passthrough();

const scoreQuestionSchema = z
  .object({
    type: z.literal("score"),
    instructions,
    criteria: z
      .array(z.string({ invalid_type_error: "Score levels must be strings." }), {
        required_error: "Score criteria must be an array of levels.",
        invalid_type_error: "Score criteria must be an array of levels.",
      })
      .min(2, "Score criteria must contain at least 2 levels."),
  })
  .passthrough();

const QUESTION_ERROR_MESSAGES: Partial<Record<z.ZodIssueCode, string>> = {
  [z.ZodIssueCode.invalid_union_discriminator]: "Question type must be choice, noul, or score.",
  [z.ZodIssueCode.invalid_type]: "Question must be an object.",
};

const questionSchema = z.discriminatedUnion("type", [choiceQuestionSchema, noulQuestionSchema, scoreQuestionSchema], {
  errorMap: (issue, context) => ({ message: QUESTION_ERROR_MESSAGES[issue.code] ?? context.defaultError }),
});

export const systemOneRequestSchema = z
  .object(
    {
      model: nonEmptyString("Model must be a non-empty string.").optional(),
      state: z.unknown(),
      questions: z
        .record(questionSchema, {
          required_error: "Required property 'questions' is missing.",
          invalid_type_error: "Questions must be a non-empty object.",
        })
        .refine((questions) => Object.keys(questions).length > 0, "At least one question is required."),
    },
    { invalid_type_error: "Payload must be a JSON object." },
  )
  .passthrough()
  .refine((request) => "state" in request, { message: "Required property 'state' is missing.", path: ["state"] });

const probability = z.number().finite().min(0).max(1);
const probabilities = z.record(probability);

const noulAnswerSchema = z.object({ type: z.literal("noul"), noul: probability }).passthrough();

const choiceAnswerShape = {
  type: z.literal("choice"),
  choice: z.string(),
  confidence: probability.optional(),
  probabilities,
};
const choiceAnswerSchema = z.object(choiceAnswerShape).passthrough();

const scoreAnswerShape = {
  type: z.literal("score"),
  score: z.number().finite(),
  confidence: probability.optional(),
  legend: z.record(z.string()).optional(),
  probabilities,
};
const scoreAnswerSchema = z.object(scoreAnswerShape).passthrough();

const tokenCount = z.number().finite().nonnegative();

export const systemOneResponseSchema = z
  .object({
    model: nonEmptyString("Model must be a non-empty string.").nullish(),
    answers: z.record(z.discriminatedUnion("type", [noulAnswerSchema, choiceAnswerSchema, scoreAnswerSchema])),
    usage: z.object({ input_tokens: tokenCount, output_tokens: tokenCount }).optional(),
  })
  .passthrough();

export type SystemOneRequest = z.infer<typeof systemOneRequestSchema>;
export type SystemOneQuestion = z.infer<typeof questionSchema>;
export type SystemOneResponse = z.infer<typeof systemOneResponseSchema>;
export type SystemOneAnswer = SystemOneResponse["answers"][string];
