import { z } from "zod";
import { parseJson } from "./validatePayload";

const choiceQuestion = z.looseObject({
  type: z.literal("choice"),
  instructions: z.string().optional(),
  criteria: z.record(z.string(), z.string()),
});

const noulQuestion = z.looseObject({
  type: z.literal("noul"),
  instructions: z.string().optional(),
  criteria: z.looseObject({ true: z.string().optional(), false: z.string().optional() }).optional(),
});

const scoreQuestion = z.looseObject({
  type: z.literal("score"),
  instructions: z.string().optional(),
  criteria: z.array(z.string()),
});

const formQuestion = z.discriminatedUnion("type", [choiceQuestion, noulQuestion, scoreQuestion]);

const formPayload = z.looseObject({
  state: z.string().optional(),
  questions: z.record(z.string(), formQuestion).optional(),
});

export type FormPayload = z.infer<typeof formPayload>;
export type FormQuestion = z.infer<typeof formQuestion>;
export type ChoiceQuestion = z.infer<typeof choiceQuestion>;
export type NoulQuestion = z.infer<typeof noulQuestion>;
export type ScoreQuestion = z.infer<typeof scoreQuestion>;
export type QuestionType = FormQuestion["type"];
export type NoulSide = "true" | "false";

const formPayloadInInputOrder = z.custom<FormPayload>((value) => formPayload.safeParse(value).success);

export const readForm = (raw: string): FormPayload | undefined => {
  const json = parseJson(raw);
  if (!json.ok) {
    return undefined;
  }
  return formPayloadInInputOrder.safeParse(json.value).data;
};

export const freeName = (taken: readonly string[], prefix: string): string =>
  Array.from({ length: taken.length + 1 }, (_, index) => `${prefix}_${index + 1}`).find(
    (name) => !taken.includes(name),
  ) ?? `${prefix}_${taken.length + 1}`;

export const renameKey = <T>(record: Readonly<Record<string, T>>, from: string, to: string): Record<string, T> =>
  Object.fromEntries(Object.entries(record).map(([key, value]) => [key === from ? to : key, value]));

export const withoutKey = <T>(record: Readonly<Record<string, T>>, key: string): Record<string, T> =>
  Object.fromEntries(Object.entries(record).filter(([entry]) => entry !== key));

export const replaceAt = <T>(items: readonly T[], index: number, value: T): T[] =>
  items.map((item, position) => (position === index ? value : item));

export const removeAt = <T>(items: readonly T[], index: number): T[] =>
  items.filter((_, position) => position !== index);

export const blankQuestion = (): ChoiceQuestion => ({
  type: "choice",
  instructions: "",
  criteria: { option_1: "", option_2: "" },
});

export const retype = (question: FormQuestion, type: QuestionType): FormQuestion => {
  if (question.type === type) {
    return question;
  }
  const { criteria: _criteria, type: _type, ...rest } = question;
  switch (type) {
    case "choice":
      return { type, ...rest, criteria: blankQuestion().criteria };
    case "noul":
      return { type, ...rest };
    case "score":
      return { type, ...rest, criteria: ["", ""] };
  }
};

export const withNoulCriterion = (question: NoulQuestion, side: NoulSide, text: string): NoulQuestion => {
  const { criteria: current, ...rest } = question;
  const criteria = Object.fromEntries(Object.entries({ ...current, [side]: text }).filter(([, value]) => value !== ""));
  return Object.keys(criteria).length === 0 ? rest : { ...rest, criteria };
};
