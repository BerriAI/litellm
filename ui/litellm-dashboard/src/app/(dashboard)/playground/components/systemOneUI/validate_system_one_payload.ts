import type { SystemOneQuestion, SystemOneRequest } from "./system_one_types";

export interface SystemOnePayloadIssue {
  path: string;
  message: string;
  severity: "error" | "warning";
}

export interface SystemOnePayloadValidation {
  isValid: boolean;
  payload?: SystemOneRequest;
  issues: SystemOnePayloadIssue[];
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const questionIssue = (
  path: string,
  message: string,
  severity: "error" | "warning" = "error",
): SystemOnePayloadIssue => ({ path, message, severity });

function validateChoiceCriteria(criteria: unknown, path: string): SystemOnePayloadIssue[] {
  if (!isRecord(criteria)) {
    return [questionIssue(path, "Choice criteria must be an object mapping labels to descriptions.")];
  }
  if (Object.keys(criteria).length < 1 || Object.keys(criteria).length > 255) {
    return [questionIssue(path, "Choice criteria must contain between 1 and 255 options.")];
  }
  return [];
}

function validateScoreCriteria(criteria: unknown, path: string): SystemOnePayloadIssue[] {
  if (!Array.isArray(criteria)) {
    return [questionIssue(path, "Score criteria must be an array of levels.")];
  }
  if (criteria.length < 2) {
    return [questionIssue(path, "Score criteria must contain at least 2 levels.")];
  }
  return criteria.length > 10
    ? [questionIssue(path, "More than 10 score levels may reduce result quality.", "warning")]
    : [];
}

function validateQuestion(id: string, value: unknown): SystemOnePayloadIssue[] {
  const path = `questions.${id}`;
  if (!isRecord(value)) {
    return [questionIssue(path, "Question must be an object.")];
  }
  if (!["choice", "noul", "score"].includes(String(value.type))) {
    return [questionIssue(`${path}.type`, "Question type must be choice, noul, or score.")];
  }

  const instructionIssues =
    "instructions" in value
      ? []
      : [questionIssue(`${path}.instructions`, "Required property 'instructions' is missing.")];

  if (value.type === "choice") {
    return [...instructionIssues, ...validateChoiceCriteria(value.criteria, `${path}.criteria`)];
  }
  if (value.type === "score") {
    return [...instructionIssues, ...validateScoreCriteria(value.criteria, `${path}.criteria`)];
  }
  if ("criteria" in value && !isRecord(value.criteria)) {
    return [
      ...instructionIssues,
      questionIssue(`${path}.criteria`, "Noul criteria, when provided, must be an object."),
    ];
  }
  return instructionIssues;
}

function validateQuestions(value: unknown): SystemOnePayloadIssue[] {
  if (!isRecord(value)) {
    return [questionIssue("questions", "Questions must be a non-empty object.")];
  }
  if (Object.keys(value).length === 0) {
    return [questionIssue("questions", "At least one question is required.")];
  }
  return Object.entries(value).flatMap(([id, question]) => validateQuestion(id, question));
}

export function validateSystemOnePayload(raw: string): SystemOnePayloadValidation {
  if (!raw.trim()) {
    return {
      isValid: false,
      issues: [{ path: "root", message: "Payload cannot be empty.", severity: "error" }],
    };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      isValid: false,
      issues: [{ path: "syntax", message: `Invalid JSON syntax: ${message}`, severity: "error" }],
    };
  }

  if (!isRecord(parsed)) {
    return {
      isValid: false,
      issues: [{ path: "root", message: "Payload must be a JSON object.", severity: "error" }],
    };
  }

  const issues: SystemOnePayloadIssue[] = [];

  if (!("state" in parsed)) {
    issues.push(questionIssue("state", "Required property 'state' is missing."));
  }

  if ("model" in parsed && (typeof parsed.model !== "string" || !parsed.model.trim())) {
    issues.push(questionIssue("model", "Model must be a non-empty string."));
  }

  if (!("questions" in parsed)) {
    issues.push(questionIssue("questions", "Required property 'questions' is missing."));
  } else {
    issues.push(...validateQuestions(parsed.questions));
  }

  const isValid = !issues.some((issue) => issue.severity === "error");
  const payload =
    isValid && isRecord(parsed.questions)
      ? ({
          ...(typeof parsed.model === "string" ? { model: parsed.model } : {}),
          state: parsed.state,
          questions: parsed.questions as Record<string, SystemOneQuestion>,
        } satisfies SystemOneRequest)
      : undefined;

  return { isValid, payload, issues };
}
