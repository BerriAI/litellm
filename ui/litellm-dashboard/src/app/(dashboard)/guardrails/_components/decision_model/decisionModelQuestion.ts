import type { DecisionModelCheckDraft } from "./buildDecisionModelParams";

export interface DecisionModelGroup {
  model_group: string;
  providers?: string[] | null;
}

export function decisionProvidersForGroups(
  allowedProviders: readonly string[],
  groups: readonly DecisionModelGroup[],
): string[] {
  const deployed = new Set(groups.flatMap((group) => group.providers ?? []));
  return allowedProviders.filter((provider) => deployed.has(provider));
}

export function decisionModelsForProvider(groups: readonly DecisionModelGroup[], provider: string | null): string[] {
  if (!provider) return [];
  return groups
    .filter((group) => (group.providers ?? []).includes(provider))
    .map((group) => group.model_group)
    .toSorted((a, b) => a.localeCompare(b));
}

export interface DecisionTestBody {
  model: string;
  input: string;
  questions: Array<{ type: "predicate"; name: string; instructions: string }>;
}

export function enabledDecisionQuestions(checks: readonly DecisionModelCheckDraft[]): DecisionModelCheckDraft[] {
  return checks.filter((check) => check.enabled !== false && check.instructions.trim().length > 0);
}

export function buildDecisionTestBody(
  model: string,
  input: string,
  checks: readonly DecisionModelCheckDraft[],
): DecisionTestBody {
  return {
    model,
    input,
    questions: enabledDecisionQuestions(checks).map((check) => ({
      type: "predicate" as const,
      name: check.name,
      instructions: check.instructions,
    })),
  };
}

export type DecisionTestResult =
  | { kind: "probability"; probability: number }
  | { kind: "refused" }
  | { kind: "missing" };

export type DecisionTestResults = Record<string, DecisionTestResult>;

export type DecisionTestRun =
  | { id: number; input: string; results: DecisionTestResults }
  | { id: number; input: string; error: string };

interface DecisionsApiAnswer {
  type?: string;
  name?: string | null;
  probability?: number;
}

export function parseDecisionTestResponse(body: unknown, questionNames: readonly string[]): DecisionTestResults {
  const answers =
    typeof body === "object" && body !== null ? (body as { answers?: DecisionsApiAnswer[] }).answers ?? [] : [];
  const byName = new Map(answers.filter((answer) => answer.name != null).map((answer) => [answer.name, answer]));
  const results: DecisionTestResults = {};
  for (const name of questionNames) {
    const answer = byName.get(name);
    if (!answer) {
      results[name] = { kind: "missing" };
    } else if (answer.type === "refusal") {
      results[name] = { kind: "refused" };
    } else if (typeof answer.probability === "number") {
      results[name] = { kind: "probability", probability: answer.probability };
    } else {
      results[name] = { kind: "missing" };
    }
  }
  return results;
}

export type DecisionTestChip = "block" | "pass" | "logged" | "no_answer";

export function decisionTestChip(result: DecisionTestResult, check: DecisionModelCheckDraft): DecisionTestChip {
  if (result.kind !== "probability") return "no_answer";
  if (result.probability >= check.threshold) return check.action === "block" ? "block" : "logged";
  return "pass";
}

export function decisionTestOverall(
  results: DecisionTestResults,
  checks: readonly DecisionModelCheckDraft[],
): "block" | "pass" {
  const byName = new Map(checks.map((check) => [check.name, check]));
  for (const [name, result] of Object.entries(results)) {
    const check = byName.get(name);
    if (check && decisionTestChip(result, check) === "block") return "block";
  }
  return "pass";
}

export function visibleTestResults(
  results: DecisionTestResults,
  checks: readonly DecisionModelCheckDraft[],
): Array<{ check: DecisionModelCheckDraft; result: DecisionTestResult }> {
  return checks.flatMap((check) => {
    const result = results[check.name];
    return result ? [{ check, result }] : [];
  });
}

export function runTestShortcutLabel(platform?: string): string {
  const hint = platform ?? (typeof navigator === "undefined" ? "" : `${navigator.platform} ${navigator.userAgent}`);
  return /mac/i.test(hint) ? "⌘+Enter" : "Ctrl+Enter";
}

export const DEFAULT_DECISION_THRESHOLD = 0.7;

export const DECISION_TEST_HISTORY_CAP = 20;

export function prependTestRun(runs: readonly DecisionTestRun[], run: DecisionTestRun): DecisionTestRun[] {
  return [run, ...runs].slice(0, DECISION_TEST_HISTORY_CAP);
}
