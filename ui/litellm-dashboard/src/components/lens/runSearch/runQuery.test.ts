import { describe, expect, it } from "vitest";

import { run, runs } from "./__fixtures__/runs";
import { fieldValues, filterRuns } from "./runQuery";

const ids = (query: string) => filterRuns(runs, query).map((r) => r.trace_id);

describe("filterRuns", () => {
  it("returns every run for an empty or blank query", () => {
    expect(filterRuns(runs, "   ")).toBe(runs);
  });

  it("matches free text against input, name and trace id, ignoring case, with every term required", () => {
    expect(ids("REFUND")).toEqual(["aaa111"]);
    expect(ids("lead")).toEqual(["bbb222"]);
    expect(ids("ccc3")).toEqual(["ccc333"]);
    expect(ids("vector research")).toEqual(["bbb222"]);
    expect(ids("vector refund")).toEqual([]);
  });

  it("keeps a quoted phrase together", () => {
    expect(ids('"my refund"')).toEqual(["aaa111"]);
    expect(ids('"refund my"')).toEqual([]);
  });

  it("splits runs by status, judged by recorded errors", () => {
    expect(ids("status:error")).toEqual(["bbb222"]);
    expect(ids("-status:error")).toEqual(["aaa111", "ccc333"]);
    expect(ids("status:OK")).toEqual(["aaa111", "ccc333"]);
  });

  it("matches a multi-valued field when any value matches, and negation when none does", () => {
    expect(ids("agent:triage")).toEqual(["aaa111"]);
    expect(ids("-agent:triage")).toEqual(["bbb222", "ccc333"]);
    expect(ids("agent:cron")).toEqual(["ccc333"]);
    expect(ids("model:gpt-5")).toEqual(["aaa111"]);
  });

  it("requires a whole-value match unless the value has wildcards", () => {
    expect(ids("agent:billing")).toEqual([]);
    expect(ids("agent:*bill*")).toEqual(["aaa111"]);
    expect(ids("agent:billing*")).toEqual(["aaa111"]);
    expect(ids("agent:*agent")).toEqual(["aaa111"]);
    expect(ids("-agent:*search*")).toEqual(["aaa111", "ccc333"]);
    expect(ids("name:res.arch_lead")).toEqual([]);
  });

  it("matches glob segments case-insensitively without overlapping exact anchors", () => {
    const billing = run({ trace_id: "billing", agent_names: ["BILLING-agent"] });
    const research = run({ trace_id: "research", agent_names: ["research-billing-agent"] });
    const doubled = run({ trace_id: "doubled", agent_names: ["agent"] });
    const globRuns = [billing, research, doubled];

    expect(filterRuns(globRuns, "agent:*bill*")).toEqual([billing, research]);
    expect(filterRuns(globRuns, "agent:billing*")).toEqual([billing]);
    expect(filterRuns(globRuns, "agent:*agent")).toEqual(globRuns);
    expect(filterRuns([run({ name: "aa" })], "name:a*a")).toHaveLength(1);
    expect(filterRuns([run({ name: "a" })], "name:a*a")).toHaveLength(0);
    expect(filterRuns(globRuns, "agent:**")).toEqual(filterRuns(globRuns, "agent:*"));
  });

  it("matches adversarial globs without catastrophic backtracking", () => {
    const longValue = run({ agent_names: ["a".repeat(5000)] });
    const pattern = `agent:${"a*".repeat(30)}b`;
    const startedAt = performance.now();
    const matches = filterRuns([longValue], pattern);
    const elapsed = performance.now() - startedAt;

    expect(matches).toEqual([]);
    expect(elapsed).toBeLessThan(100);
  });

  it("searches the readable input text and quoted values with spaces", () => {
    expect(ids('input:"*vector stores"')).toEqual(["bbb222"]);
    expect(ids("trace_id:bbb222")).toEqual(["bbb222"]);
  });

  it("combines field clauses with free text", () => {
    expect(ids("agent:*e* -status:error refund")).toEqual(["aaa111"]);
  });

  it("ignores a key still waiting for its value and treats unknown keys as text", () => {
    expect(ids("agent:")).toEqual(["aaa111", "bbb222", "ccc333"]);
    expect(ids("refund agent:")).toEqual(["aaa111"]);
    expect(ids("foo:bar")).toEqual([]);
    expect(filterRuns([run({ name: "foo:bar" })], "foo:bar")).toHaveLength(1);
  });
});

describe("fieldValues", () => {
  it("lists each distinct non-empty value once, sorted", () => {
    expect(fieldValues(runs, "agent")).toEqual(["billing-agent", "cron", "researcher", "triage"]);
    expect(fieldValues(runs, "status")).toEqual(["error", "ok"]);
    expect(fieldValues([run({ models: [] })], "model")).toEqual([]);
  });
});
