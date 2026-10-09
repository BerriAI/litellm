import { describe, expect, it } from "vitest";

import { caseReplySummary, EMPTY_DRAFT, revisionCases, setExpected, toggleCase } from "./draft";
import type { DatasetCase } from "./types";

const source = { trace_id: "t", trace_ref: "", span_id: "", finding_id: "", lens_id: "" };
const makeCase = (id: string, overrides: Partial<DatasetCase> = {}): DatasetCase => ({
  id,
  messages: [{ role: "user", content: `question ${id}`, name: "", tool_calls: [] }],
  reply: `reply ${id}`,
  tool_calls: [],
  expected: "",
  included: true,
  source,
  agent_version: "",
  ...overrides,
});

describe("revisionCases", () => {
  it("keeps every existing case first and appends only the ticked new ones with their edited expected", () => {
    const existing = [makeCase("old", { expected: "kept" })];
    const built = [makeCase("a"), makeCase("b"), makeCase("c")];
    const draft = setExpected(toggleCase(EMPTY_DRAFT, "b"), "c", "should refund");
    expect(revisionCases(existing, built, draft).map((item) => [item.id, item.expected])).toEqual([
      ["old", "kept"],
      ["a", ""],
      ["c", "should refund"],
    ]);
  });

  it("re-ticking a case puts it back", () => {
    const built = [makeCase("a")];
    expect(revisionCases([], built, toggleCase(toggleCase(EMPTY_DRAFT, "a"), "a"))).toHaveLength(1);
  });
});

describe("caseReplySummary", () => {
  it("shows tool calls when the agent only called tools", () => {
    expect(
      caseReplySummary(makeCase("a", { reply: "", tool_calls: [{ name: "lookup", arguments: '{"id":1}' }] })),
    ).toBe('lookup({"id":1})');
  });
});
