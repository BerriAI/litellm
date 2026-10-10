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

export function buildDecisionTestBody(
  model: string,
  input: string,
  name: string,
  instructions: string,
): DecisionTestBody {
  return {
    model,
    input,
    questions: [{ type: "predicate", name, instructions }],
  };
}

export type DecisionTestVerdict =
  | { kind: "probability"; probability: number }
  | { kind: "refused" }
  | { kind: "error"; message: string };

interface DecisionsApiAnswer {
  type?: string;
  name?: string | null;
  probability?: number;
}

export function parseDecisionTestResponse(body: unknown, questionName: string): DecisionTestVerdict {
  if (typeof body !== "object" || body === null) {
    return { kind: "error", message: "Unexpected response from the decision model" };
  }
  const answers = (body as { answers?: DecisionsApiAnswer[] }).answers;
  if (!Array.isArray(answers)) {
    const message = (body as { error?: { message?: string } }).error?.message;
    return { kind: "error", message: message ?? "Unexpected response from the decision model" };
  }
  const answer = answers.find((entry) => entry.name === questionName) ?? answers[0];
  if (!answer) {
    return { kind: "error", message: "The decision model returned no answers" };
  }
  if (answer.type === "refusal") {
    return { kind: "refused" };
  }
  if (typeof answer.probability !== "number") {
    return { kind: "error", message: "The decision model returned no probability" };
  }
  return { kind: "probability", probability: answer.probability };
}

export type DecisionTestOutcome = "would_block" | "would_log" | "pass" | "refused" | "error";

export function decisionTestOutcome(
  verdict: DecisionTestVerdict,
  action: "block" | "log",
  threshold: number,
): DecisionTestOutcome {
  if (verdict.kind === "refused") return "refused";
  if (verdict.kind === "error") return "error";
  if (verdict.probability >= threshold) return action === "block" ? "would_block" : "would_log";
  return "pass";
}
