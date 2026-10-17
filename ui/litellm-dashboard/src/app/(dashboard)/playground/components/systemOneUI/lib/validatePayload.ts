import { z } from "zod";
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

const error = (path: readonly PropertyKey[], message: string): SystemOnePayloadIssue => ({
  path: path.length > 0 ? path.join(".") : "root",
  label: issueLabel(path),
  message,
  severity: "error",
});

const toIssue = (path: PropertyKey[], message: string): SystemOnePayloadIssue =>
  path[0] === "questions" && path[1] === ""
    ? error(["questions"], "Give every question a name.")
    : error(path, message);

const invalid = (path: string, message: string) => ({
  isValid: false,
  issues: [error([path], message)],
});

export function parseJson(raw: string): { ok: true; value: unknown } | { ok: false; message: string } {
  try {
    return { ok: true, value: JSON.parse(raw) };
  } catch (cause: unknown) {
    return { ok: false, message: cause instanceof Error ? cause.message : String(cause) };
  }
}

const scoreLevelWarnings = (payload: PlaygroundRequest): SystemOnePayloadIssue[] =>
  Object.entries(payload.questions)
    .filter(([, question]) => question.type === "score" && question.criteria.length > RECOMMENDED_MAX_SCORE_LEVELS)
    .map(([id]) => ({
      path: `questions.${id}.criteria`,
      label: issueLabel(["questions", id, "criteria"]),
      message: `More than ${RECOMMENDED_MAX_SCORE_LEVELS} score levels may reduce result quality.`,
      severity: "warning",
    }));

function validatePayload<T>(raw: string, schema: z.ZodType<T>): PayloadValidation<T> {
  if (!raw.trim()) {
    return invalid("root", "Payload cannot be empty.");
  }

  const json = parseJson(raw);
  if (!json.ok) {
    return invalid("syntax", `Invalid JSON syntax: ${json.message}`);
  }

  const result = schema.safeParse(json.value);
  if (!result.success) {
    return {
      isValid: false,
      issues: result.error.issues.map((issue) => toIssue(issue.path, issue.message)),
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

const requireContext = <T>(raw: string, path: "state" | "input", result: PayloadValidation<T>): PayloadValidation<T> =>
  isMissingContext(raw, path)
    ? {
        ...result,
        isValid: false,
        issues: [error([path], "Add the text the model should decide about.")],
      }
    : result;

export function validateSystemOnePayload(
  raw: string,
  endpoint: DecisionEndpoint = "/typesafe/v1/systemone",
): SystemOnePayloadValidation {
  const schema = endpoint === "/v1/systemone" ? decisionsRequestSchema : systemOneRequestSchema;
  const result = validatePayload<PlaygroundRequest>(raw, schema);
  return requireContext(
    raw,
    "state",
    result.payload ? { ...result, issues: scoreLevelWarnings(result.payload) } : result,
  );
}

export function validateOpenAIDecisionsPayload(raw: string): PayloadValidation<OpenAIDecisionsRequest> {
  const result = validatePayload<OpenAIDecisionsRequest>(raw, openAIDecisionsRequestSchema);
  return requireContext(raw, "input", result);
}
