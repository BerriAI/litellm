export interface DecisionModelCheckDraft {
  name: string;
  label?: string;
  instructions?: string;
  action: "block" | "log";
  threshold: number;
  custom?: boolean;
}

export interface DecisionModelLitellmParams {
  decision_model: string;
  checks: Array<{
    name: string;
    instructions?: string;
    action: "block" | "log";
    threshold: number;
  }>;
}

const clampThreshold = (value: number): number => Math.min(1, Math.max(0, value));

export function buildDecisionModelParams(
  decisionModel: string,
  checks: readonly DecisionModelCheckDraft[],
): DecisionModelLitellmParams {
  return {
    decision_model: decisionModel,
    checks: checks.map((check) => ({
      name: check.name,
      // Preset checks omit instructions so the backend resolves the canonical preset text;
      // custom checks must carry their own.
      ...(check.custom ? { instructions: check.instructions ?? "" } : {}),
      action: check.action,
      threshold: clampThreshold(check.threshold),
    })),
  };
}
