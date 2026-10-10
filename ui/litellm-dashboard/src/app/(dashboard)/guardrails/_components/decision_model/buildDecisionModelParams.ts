export interface DecisionModelCheckDraft {
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
  return checks.filter((check) => check.enabled !== false);
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
