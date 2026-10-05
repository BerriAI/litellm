import type { Finding, IssueBrief } from "./types";

export function sortedFindings(findings: Finding[]): Finding[] {
  const rank = { high: 0, medium: 1, low: 2 };
  return [...findings].sort(
    (a, b) =>
      rank[a.priority ?? "medium"] - rank[b.priority ?? "medium"] || Date.parse(b.last_seen) - Date.parse(a.last_seen),
  );
}

export interface EvidenceTarget {
  readonly id: string;
  readonly traceRef: string;
}

const TRACE_REF_LENGTH = 64;

/** An execution id is `<trace_ref>:<trace_id>`; ids saved before that format no longer resolve. */
export function evidenceTarget(executionId: string): EvidenceTarget | null {
  const traceRef = executionId.slice(0, TRACE_REF_LENGTH);
  const id = executionId.slice(TRACE_REF_LENGTH + 1);
  if (traceRef.length !== TRACE_REF_LENGTH || executionId[TRACE_REF_LENGTH] !== ":" || !id) return null;
  return { id, traceRef };
}

export function briefMarkdown(title: string, brief: IssueBrief): string {
  return [
    `# ${title}`,
    `## Problem\n${brief.problem}`,
    `## User goal\n${brief.user_goal}`,
    `## What happened\n${brief.what_happened}`,
    `## Test cases\n${brief.test_cases.map((t, i) => `${i + 1}. **Input:** ${t.input}  \n   **Expect:** ${t.expected}`).join("\n")}`,
  ].join("\n\n");
}

export function mergeFeedback(findings: Finding[], current: Finding[]): Finding[] {
  return findings.map((finding) => {
    const feedback = current.find((item) => item.id === finding.id);
    return feedback ? { ...finding, status: feedback.status, reason: feedback.reason } : finding;
  });
}
