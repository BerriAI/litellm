import type { TFunction } from "i18next";
import { z } from "zod";
import i18n from "@/i18n/i18n";
import { issueLabel } from "./issueLabel";
import { openAIDecisionsRequestSchema, type OpenAIDecisionsRequest } from "./openAIDecisions";
import {
  decisionsRequestSchema,
  systemOneRequestSchema,
  type DecisionEndpoint,
  type PlaygroundRequest,
} from "./schemas";

const RECOMMENDED_MAX_SCORE_LEVELS = 10;

export interface SystemOnePayloadIssue {
  path: string;
  label: string;
  message: string;
  severity: "error" | "warning";
}

export interface PayloadValidation<T> {
  isValid: boolean;
  payload?: T;
  issues: SystemOnePayloadIssue[];
}

export type SystemOnePayloadValidation = PayloadValidation<PlaygroundRequest>;

export const errorCount = (validation: PayloadValidation<unknown>): number =>
  validation.issues.filter((issue) => issue.severity === "error").length;

const error = (path: readonly PropertyKey[], message: string, t: TFunction): SystemOnePayloadIssue => ({
  path: path.length > 0 ? path.join(".") : "root",
  label: issueLabel(path, t),
  message,
  severity: "error",
});

const toIssue = (path: PropertyKey[], message: string, t: TFunction): SystemOnePayloadIssue =>
  path[0] === "questions" && path[1] === ""
    ? error(["questions"], t("playground.systemOne.validatePayload.nameEveryQuestion"), t)
    : error(path, message, t);

const invalid = (path: string, message: string, t: TFunction) => ({
  isValid: false,
  issues: [error([path], message, t)],
});

export function parseJson(raw: string): { ok: true; value: unknown } | { ok: false; message: string } {
  try {
    return { ok: true, value: JSON.parse(raw) };
  } catch (cause: unknown) {
    return { ok: false, message: cause instanceof Error ? cause.message : String(cause) };
  }
}

const scoreLevelWarnings = (payload: PlaygroundRequest, t: TFunction): SystemOnePayloadIssue[] =>
  Object.entries(payload.questions)
    .filter(([, question]) => question.type === "score" && question.criteria.length > RECOMMENDED_MAX_SCORE_LEVELS)
    .map(([id]) => ({
      path: `questions.${id}.criteria`,
      label: issueLabel(["questions", id, "criteria"], t),
      message: t("playground.systemOne.validatePayload.tooManyLevels", { max: RECOMMENDED_MAX_SCORE_LEVELS }),
      severity: "warning",
    }));

function validatePayload<T>(raw: string, schema: z.ZodType<T>, t: TFunction): PayloadValidation<T> {
  if (!raw.trim()) {
    return invalid("root", t("playground.systemOne.validatePayload.empty"), t);
  }

  const json = parseJson(raw);
  if (!json.ok) {
    return invalid("syntax", t("playground.systemOne.validatePayload.invalidJson", { message: json.message }), t);
  }

  const result = schema.safeParse(json.value);
  if (!result.success) {
    return {
      isValid: false,
      issues: result.error.issues.map((issue) => toIssue(issue.path, issue.message, t)),
    };
  }

  return { isValid: true, payload: result.data, issues: [] };
}

const isBlankContext = (context: unknown): boolean =>
  (typeof context === "string" && context.trim() === "") || (Array.isArray(context) && context.length === 0);

const isMissingContext = (raw: string, path: "state" | "input"): boolean => {
  const json = parseJson(raw);
  if (!json.ok || typeof json.value !== "object" || json.value === null) {
    return false;
  }
  return isBlankContext((json.value as Record<string, unknown>)[path]);
};

const requireContext = <T>(
  raw: string,
  path: "state" | "input",
  result: PayloadValidation<T>,
  t: TFunction,
): PayloadValidation<T> =>
  isMissingContext(raw, path)
    ? {
        ...result,
        isValid: false,
        issues: [error([path], t("playground.systemOne.validatePayload.missingContext"), t)],
      }
    : result;

export function validateSystemOnePayload(
  raw: string,
  endpoint: DecisionEndpoint = "/typesafe/v1/systemone",
  t: TFunction = i18n.t.bind(i18n),
): SystemOnePayloadValidation {
  const schema = endpoint === "/v1/systemone" ? decisionsRequestSchema : systemOneRequestSchema;
  const result = validatePayload<PlaygroundRequest>(raw, schema, t);
  return requireContext(
    raw,
    "state",
    result.payload ? { ...result, issues: scoreLevelWarnings(result.payload, t) } : result,
    t,
  );
}

export function validateOpenAIDecisionsPayload(
  raw: string,
  t: TFunction = i18n.t.bind(i18n),
): PayloadValidation<OpenAIDecisionsRequest> {
  const result = validatePayload<OpenAIDecisionsRequest>(raw, openAIDecisionsRequestSchema, t);
  return requireContext(raw, "input", result, t);
}
