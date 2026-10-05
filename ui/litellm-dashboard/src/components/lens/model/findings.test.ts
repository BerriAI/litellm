import { mergeFeedback, sortedFindings, briefMarkdown } from "./findings";
import { describe, expect, it } from "vitest";
import { type Finding } from "./types";

describe("Lens findings", () => {
  it("puts high priority issues ahead of newer low priority findings", () => {
    const base: Finding = {
      kind: "issue",
      status: "open",
      reason: "",
      suggestion: "",
      limitation: "",
      occurrences: [],
      id: "low",
      check_id: "check",
      title: "Recovered error",
      description: "The run recovered.",
      evidence: [],
      revision: 1,
      priority: "low",
      first_seen: "2026-09-30T10:00:00Z",
      last_seen: "2026-09-30T12:00:00Z",
    };
    const high: Finding = { ...base, id: "high", priority: "high", last_seen: "2026-09-30T11:00:00Z" };
    expect(sortedFindings([base, high]).map((f) => f.id)).toEqual(["high", "low"]);
  });
  it("turns an issue brief into a pasteable markdown document", () => {
    const briefInput = {
      problem: "The workspace was not a Git repository.",
      user_goal: "Open a PR fixing a typo",
      what_happened: 'Git returned "fatal: not a git repository"',
      test_cases: [
        { input: "Fix the typo and open a PR", expected: "A PR URL is returned" },
        { input: "Rename greet", expected: "The rename is committed" },
      ],
    };
    expect(briefMarkdown("PRs were never opened", briefInput)).toBe(
      [
        "# PRs were never opened",
        "## Problem\nThe workspace was not a Git repository.",
        "## User goal\nOpen a PR fixing a typo",
        '## What happened\nGit returned "fatal: not a git repository"',
        "## Test cases\n1. **Input:** Fix the typo and open a PR  \n   **Expect:** A PR URL is returned\n" +
          "2. **Input:** Rename greet  \n   **Expect:** The rename is committed",
      ].join("\n\n"),
    );
  });
});

it("applies current feedback to a historical snapshot while preserving its original evidence", () => {
  const snapshot: Finding = {
    kind: "issue",
    status: "open",
    reason: "",
    suggestion: "",
    limitation: "",
    occurrences: [],
    id: "finding",
    check_id: "check",
    title: "Historical title",
    description: "Historical description",
    evidence: [],
    revision: 1,
    priority: "high",
    first_seen: "2026-09-30T10:00:00Z",
    last_seen: "2026-09-30T12:00:00Z",
  };
  const current: Finding = { ...snapshot, title: "Updated title", status: "resolved", reason: "Fixed", revision: 2 };
  expect(mergeFeedback([snapshot], [current])).toEqual([{ ...snapshot, status: "resolved", reason: "Fixed" }]);
  expect(snapshot.status).toBe("open");
  expect(mergeFeedback([snapshot], [])).toEqual([snapshot]);
});
