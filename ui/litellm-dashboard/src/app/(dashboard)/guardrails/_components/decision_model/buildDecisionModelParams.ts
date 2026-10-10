export interface DecisionModelCheckDraft {
  id: string;
  name: string;
  instructions: string;
  action: "block" | "log";
  threshold: number;
  enabled?: boolean;
}

export interface DecisionModelLitellmParams {
  decision_model: string;
  checks: Array<{
    name: string;
    instructions: string;
    action: "block" | "log";
    threshold: number;
  }>;
}

export function enabledDecisionChecks(checks: readonly DecisionModelCheckDraft[]): readonly DecisionModelCheckDraft[] {
  return checks
    .filter((check) => check.enabled !== false)
    .map((check) => ({ ...check, name: check.name.trim(), instructions: check.instructions.trim() }));
}

export function duplicateDecisionCheckNames(checks: readonly DecisionModelCheckDraft[]): ReadonlySet<string> {
  const names = enabledDecisionChecks(checks)
    .map((check) => check.name)
    .filter(Boolean);
  return new Set(names.filter((name, index) => names.indexOf(name) !== index));
}

export function decisionChecksProblem(checks: readonly DecisionModelCheckDraft[]): string | null {
  const enabled = enabledDecisionChecks(checks);
  if (enabled.length === 0) return "Add at least one question";
  if (enabled.some((check) => !check.name || !check.instructions)) {
    return "Give every question a name and a question, or remove it";
  }
  if (duplicateDecisionCheckNames(checks).size > 0) return "Each question needs a different name";
  return null;
}

const clampThreshold = (value: number): number => Math.min(1, Math.max(0, value));

export function buildDecisionModelParams(
  decisionModel: string,
  checks: readonly DecisionModelCheckDraft[],
): DecisionModelLitellmParams {
  return {
    decision_model: decisionModel,
    checks: enabledDecisionChecks(checks).map((check) => ({
      name: check.name,
      instructions: check.instructions,
      action: check.action,
      threshold: clampThreshold(check.threshold),
    })),
  };
}
