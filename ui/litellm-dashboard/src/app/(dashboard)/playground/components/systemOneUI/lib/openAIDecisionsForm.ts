import { z } from "zod";
import { openAIDecisionsRequestSchema } from "./openAIDecisions";
import { parseJson } from "./validatePayload";

const questionFields = { name: z.string().nullish(), instructions: z.string().optional() };
const choiceFields = { value: z.union([z.string(), z.boolean()]), description: z.string().nullish() };
const levelFields = { label: z.string(), description: z.string().nullish() };
const formQuestion = z.discriminatedUnion("type", [
  z.looseObject({ ...questionFields, type: z.literal("predicate") }),
  z.looseObject({
    ...questionFields,
    type: z.literal("choice"),
    choices: z.array(z.looseObject(choiceFields)).optional(),
  }),
  z.looseObject({
    ...questionFields,
    type: z.literal("score"),
    levels: z.array(z.looseObject(levelFields)).optional(),
  }),
]);
const formPayload = z.looseObject({
  input: openAIDecisionsRequestSchema.shape.input.optional(),
  questions: z.array(formQuestion).optional(),
  safety_identifier: z.string().nullish(),
});

export type DecisionsFormPayload = z.infer<typeof formPayload>;
export type DecisionsFormQuestion = z.infer<typeof formQuestion>;
export type DecisionsQuestionType = DecisionsFormQuestion["type"];

const formPayloadInInputOrder = z.custom<DecisionsFormPayload>((value) => formPayload.safeParse(value).success);

export function readDecisionsForm(raw: string): DecisionsFormPayload | undefined {
  const json = parseJson(raw);
  return json.ok ? formPayloadInInputOrder.safeParse(json.value).data : undefined;
}

export function retypeDecisionQuestion(
  question: DecisionsFormQuestion,
  type: DecisionsQuestionType,
): DecisionsFormQuestion {
  if (question.type === type) {
    return question;
  }
  const { type: _type, choices: _choices, levels: _levels, ...rest } = question;
  switch (type) {
    case "predicate":
      return { ...rest, type };
    case "choice":
      return { ...rest, type, choices: [{ value: "option_1" }, { value: "option_2" }] };
    case "score":
      return { ...rest, type, levels: [{ label: "0" }, { label: "1" }] };
  }
}
