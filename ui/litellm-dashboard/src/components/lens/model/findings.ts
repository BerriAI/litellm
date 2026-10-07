import type { Finding, IssueBrief } from "./types";

export function sortedFindings(findings: Finding[]): Finding[] {
  const rank = { high: 0, medium: 1, low: 2 };
  return [...findings].sort(
    (a, b) =>
      rank[a.priority ?? "medium"] - rank[b.priority ?? "medium"] || Date.parse(b.last_seen) - Date.parse(a.last_seen),
  );
}

export interface EvidenceTarget {
  readonly source: string;
  readonly team: string;
  readonly id: string;
  readonly traceRef?: string;
}

export function evidenceTarget(id: string): EvidenceTarget | null {
  try {
    const parsed: unknown = JSON.parse(atob(id.replace(/-/g, "+").replace(/_/g, "/")));
    if (!Array.isArray(parsed) || ![3, 4].includes(parsed.length) || !parsed.every((item) => typeof item === "string"))
      return null;
    return { source: parsed[0], team: parsed[1], id: parsed[2], ...(parsed[3] ? { traceRef: parsed[3] } : {}) };
  } catch {
    return null;
  }
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

export function findingMarkdown(finding: Finding): string {
  if (finding.brief) return briefMarkdown(finding.title, finding.brief);
  const quotes = finding.evidence.map((e) => `> ${e.quote.replaceAll("\n", "\n> ")}`);
  return [
    `# ${finding.title}`,
    `## Summary\n${finding.description}`,
    finding.suggestion && `## Suggested fix\n${finding.suggestion}`,
    quotes.length > 0 && `## Evidence\n${quotes.join("\n\n")}`,
  ]
    .filter(Boolean)
    .join("\n\n");
}

export function mergeFeedback(findings: Finding[], current: Finding[]): Finding[] {
  return findings.map((finding) => {
    const feedback = current.find((item) => item.id === finding.id);
    return feedback ? { ...finding, status: feedback.status, reason: feedback.reason } : finding;
  });
}
