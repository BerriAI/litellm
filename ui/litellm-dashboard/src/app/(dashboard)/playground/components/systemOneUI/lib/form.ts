import { z } from "zod";
import type { DecisionEndpoint } from "./schemas";

export type FormQuestionType = "choice" | "yes_no" | "score";

export interface FormChoice {
  value: string;
  description: string;
}

export interface FormLevel {
  label: string;
  description: string;
}

export interface FormQuestion {
  type: FormQuestionType;
  name: string;
  instructions: string;
  choices: FormChoice[];
  levels: FormLevel[];
  yes: string;
  no: string;
}

export interface DecisionsForm {
  model?: string;
  context: string;
  questions: FormQuestion[];
}

export type FormFromJson = { ok: true; form: DecisionsForm } | { ok: false; reason: string };

export const isNativeEndpoint = (endpoint: DecisionEndpoint): boolean => endpoint !== "/v1/decisions";

const EMPTY_QUESTION: FormQuestion = {
  type: "choice",
  name: "",
  instructions: "",
  choices: [],
  levels: [],
  yes: "",
  no: "",
};

export function newQuestion(type: FormQuestionType): FormQuestion {
  return withQuestionType(EMPTY_QUESTION, type);
}

export function withQuestionType(question: FormQuestion, type: FormQuestionType): FormQuestion {
  const choices =
    type === "choice" && question.choices.length === 0 ? [blankChoice(), blankChoice()] : question.choices;
  const levels = type === "score" && question.levels.length === 0 ? [blankLevel(), blankLevel()] : question.levels;
  return { ...question, type, choices, levels };
}

export const blankChoice = (): FormChoice => ({ value: "", description: "" });
export const blankLevel = (): FormLevel => ({ label: "", description: "" });

const text = z.string();
const optionalText = z.string().nullish();
const optionalModel = z.string().optional();

const openAIFormSchema = z.strictObject({
  model: optionalModel,
  input: text,
  questions: z.array(
    z.strictObject({
      type: z.enum(["choice", "predicate", "score"]),
      name: optionalText,
      instructions: optionalText,
      choices: z
        .array(z.strictObject({ value: z.union([z.string(), z.boolean()]), description: optionalText }))
        .optional(),
      levels: z.array(z.strictObject({ label: text, description: optionalText })).optional(),
    }),
  ),
});

const nativeFormSchema = z.strictObject({
  model: optionalModel,
  state: text,
  questions: z.record(
    text,
    z.strictObject({
      type: z.enum(["choice", "noul", "score"]),
      instructions: optionalText,
      criteria: z.union([z.record(text, optionalText), z.array(text)]).nullish(),
    }),
  ),
});

const formJsonSchema = z.union([openAIFormSchema, nativeFormSchema]);

export function formFromJson(raw: string): FormFromJson {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    return { ok: false, reason: `Invalid JSON: ${error instanceof Error ? error.message : String(error)}` };
  }
  const result = formJsonSchema.safeParse(parsed);
  if (!result.success) {
    return { ok: false, reason: describeUnsupported(parsed, result.error) };
  }
  const payload = result.data;
  if ("input" in payload) {
    return {
      ok: true,
      form: {
        model: payload.model,
        context: payload.input,
        questions: payload.questions.map((question) => ({
          ...EMPTY_QUESTION,
          type: question.type === "predicate" ? "yes_no" : question.type,
          name: question.name ?? "",
          instructions: question.instructions ?? "",
          choices: (question.choices ?? []).map((choice) => ({
            value: String(choice.value),
            description: choice.description ?? "",
          })),
          levels: (question.levels ?? []).map((level) => ({
            label: level.label,
            description: level.description ?? "",
          })),
        })),
      },
    };
  }
  return {
    ok: true,
    form: {
      model: payload.model,
      context: payload.state,
      questions: Object.entries(payload.questions).map(([name, question]) => {
        const criteria = question.criteria ?? undefined;
        const record = criteria !== undefined && !Array.isArray(criteria) ? criteria : {};
        return {
          ...EMPTY_QUESTION,
          type: question.type === "noul" ? "yes_no" : question.type,
          name,
          instructions: question.instructions ?? "",
          choices:
            question.type === "choice"
              ? Object.entries(record).map(([value, description]) => ({ value, description: description ?? "" }))
              : [],
          levels: Array.isArray(criteria) ? criteria.map((label) => ({ label, description: "" })) : [],
          yes: question.type === "noul" ? record.true ?? "" : "",
          no: question.type === "noul" ? record.false ?? "" : "",
        };
      }),
    },
  };
}

function describeUnsupported(parsed: unknown, error: z.ZodError): string {
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return "The form needs a JSON object with input or state and questions";
  }
  const shapeIssues = error.issues
    .filter((issue) => issue.code === "invalid_union")
    .flatMap((issue) => issue.errors.flat());
  const issues = shapeIssues.length > 0 ? shapeIssues : error.issues;
  const extraKeys = issues
    .filter((issue) => issue.code === "unrecognized_keys")
    .flatMap((issue) => issue.keys.map((key) => [...issue.path, key].join(".")));
  if (extraKeys.length > 0) {
    return `The form can't show these fields: ${[...new Set(extraKeys)].join(", ")}`;
  }
  const first = issues.find((issue) => issue.path.length > 0) ?? issues[0];
  const path = first?.path.map(String).join(".") || "payload";
  return `The form can't show ${path} (${first?.message ?? "unsupported shape"}). Only string input/state, string instructions, and plain choices and levels are supported`;
}

export function formToPayload(form: DecisionsForm, endpoint: DecisionEndpoint): Record<string, unknown> {
  const model = form.model === undefined ? {} : { model: form.model };
  if (!isNativeEndpoint(endpoint)) {
    return {
      ...model,
      input: form.context,
      questions: form.questions.map((question) => ({
        type: question.type === "yes_no" ? "predicate" : question.type,
        ...(question.name ? { name: question.name } : {}),
        instructions: question.instructions,
        ...(question.type === "choice"
          ? {
              choices: question.choices.map((choice) => ({
                value: choice.value,
                ...(choice.description ? { description: choice.description } : {}),
              })),
            }
          : {}),
        ...(question.type === "score"
          ? {
              levels: question.levels.map((level) => ({
                label: level.label,
                ...(level.description ? { description: level.description } : {}),
              })),
            }
          : {}),
      })),
    };
  }
  return {
    ...model,
    state: form.context,
    questions: Object.fromEntries(
      form.questions.map((question) => [
        question.name,
        {
          type: question.type === "yes_no" ? "noul" : question.type,
          instructions: question.instructions,
          ...nativeCriteria(question),
        },
      ]),
    ),
  };
}

function nativeCriteria(question: FormQuestion): { criteria?: Record<string, string> | string[] } {
  switch (question.type) {
    case "choice":
      return { criteria: Object.fromEntries(question.choices.map((choice) => [choice.value, choice.description])) };
    case "score":
      return { criteria: question.levels.map((level) => level.label) };
    case "yes_no": {
      const criteria = {
        ...(question.yes ? { true: question.yes } : {}),
        ...(question.no ? { false: question.no } : {}),
      };
      return Object.keys(criteria).length > 0 ? { criteria } : {};
    }
  }
}

export const formToJson = (form: DecisionsForm, endpoint: DecisionEndpoint): string =>
  JSON.stringify(formToPayload(form, endpoint), null, 2);

export function formIssues(form: DecisionsForm, endpoint: DecisionEndpoint): string[] {
  const native = isNativeEndpoint(endpoint);
  const names = form.questions.map((question) => question.name.trim()).filter(Boolean);
  const duplicates = [...new Set(names.filter((name, index) => names.indexOf(name) !== index))];
  const perQuestion = form.questions.flatMap((question, index) => {
    const label = `Question ${index + 1}`;
    const values = question.type === "choice" ? question.choices.map((choice) => choice.value.trim()) : [];
    const duplicateValues = [...new Set(values.filter((value, i) => value && values.indexOf(value) !== i))];
    return [
      ...(native && !question.name.trim() ? [`${label} needs a name on ${endpoint}`] : []),
      ...(question.type === "choice" && values.some((value) => !value)
        ? [`${label} has a choice without a value`]
        : []),
      ...duplicateValues.map((value) => `${label} lists the choice "${value}" twice`),
      ...(question.type === "score" && question.levels.some((level) => !level.label.trim())
        ? [`${label} has a level without a label`]
        : []),
    ];
  });
  return [...duplicates.map((name) => `Two questions are named "${name}"`), ...perQuestion];
}
