import { z } from "zod";
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
  message: string;
  severity: "error" | "warning";
}

export interface PayloadValidation<T> {
  isValid: boolean;
  payload?: T;
  issues: SystemOnePayloadIssue[];
}

export type SystemOnePayloadValidation = PayloadValidation<PlaygroundRequest>;

const invalid = (path: string, message: string) => ({
  isValid: false,
  issues: [{ path, message, severity: "error" as const }],
});

export function parseJson(raw: string): { ok: true; value: unknown } | { ok: false; message: string } {
  try {
    return { ok: true, value: JSON.parse(raw) };
  } catch (error: unknown) {
    return { ok: false, message: error instanceof Error ? error.message : String(error) };
  }
}

const scoreLevelWarnings = (payload: PlaygroundRequest): SystemOnePayloadIssue[] =>
  Object.entries(payload.questions)
    .filter(([, question]) => question.type === "score" && question.criteria.length > RECOMMENDED_MAX_SCORE_LEVELS)
    .map(([id]) => ({
      path: `questions.${id}.criteria`,
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
      issues: result.error.issues.map((issue) => ({
        path: issue.path.length > 0 ? issue.path.join(".") : "root",
        message: issue.message,
        severity: "error",
      })),
    };
  }

  return { isValid: true, payload: result.data, issues: [] };
}

export function validateSystemOnePayload(
  raw: string,
  endpoint: DecisionEndpoint = "/typesafe/v1/systemone",
): SystemOnePayloadValidation {
  const schema = endpoint === "/v1/systemone" ? decisionsRequestSchema : systemOneRequestSchema;
  const result = validatePayload<PlaygroundRequest>(raw, schema);
  return result.payload ? { ...result, issues: scoreLevelWarnings(result.payload) } : result;
}

export function validateOpenAIDecisionsPayload(raw: string): PayloadValidation<OpenAIDecisionsRequest> {
  return validatePayload(raw, openAIDecisionsRequestSchema);
}
