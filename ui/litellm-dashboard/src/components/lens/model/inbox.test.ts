import { describe, expect, it } from "vitest";

import {
  ALL_AGENTS,
  UNKNOWN_AGENT,
  filterInbox,
  findingAgents,
  inboxAgents,
  inboxRows,
  modelsUsed,
  scheduleLabel,
  stepLine,
  type Step,
} from "./inbox";
import type { Finding, Lens } from "./types";

const finding = (overrides: Partial<Finding>): Finding =>
  ({
    id: "f",
    check_id: "c",
    kind: "issue",
    status: "open",
    title: "Tool errors swallowed",
    description: "",
    suggestion: "Retry on 5xx",
    priority: "medium",
    occurrences: ["run-1"],
    evidence: [],
    first_seen: "2026-10-01T00:00:00Z",
    last_seen: "2026-10-01T00:00:00Z",
    revision: 1,
    ...overrides,
  }) as Finding;

const lens = (
  id: string,
  agent: string,
  findings: Finding[],
  { settings = {}, runs = [] }: { settings?: Partial<Lens["settings"]>; runs?: { id: string; service: string }[] } = {},
): Lens =>
  ({
    id,
    findings,
    jobs: [{ sample: { executions: runs } }],
    next_run_at: "2026-10-03T12:10:00Z",
    settings: { name: id, agent_name: agent, service: "", enabled: true, interval_minutes: 15, ...settings },
  }) as unknown as Lens;

describe("findingAgents", () => {
  it("names the agents the finding's runs were actually recorded under", () => {
    const seen = lens("a", "", [], {
      runs: [
        { id: "run-1", service: "support-bot" },
        { id: "run-2", service: "billing-bot" },
      ],
    });
    expect(findingAgents(seen, finding({ occurrences: ["run-2", "run-1"] }))).toEqual(["billing-bot", "support-bot"]);
  });

  it("falls back to the configured agent, then to an explicit unknown, when no run says", () => {
    expect(findingAgents(lens("a", "support", []), finding({}))).toEqual(["support"]);
    expect(findingAgents(lens("a", "", []), finding({}))).toEqual([UNKNOWN_AGENT]);
  });
});

const ranked = (id: string, title: string, priority: Finding["priority"], lastSeen: string): Partial<Finding> => ({
  id,
  title,
  priority,
  last_seen: lastSeen,
});

describe("inboxRows", () => {
  it("merges the same problem for the same agent across investigations into one row", () => {
    const rows = inboxRows([
      lens("a", "support", [finding({ id: "1", occurrences: ["run-1", "run-2"] })]),
      lens("b", "support", [finding({ id: "2", title: "tool errors swallowed ", occurrences: ["run-2", "run-3"] })]),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0].sources).toHaveLength(2);
    expect(rows[0].runs).toBe(3);
  });

  it("keeps the same title for different agents as separate problems", () => {
    expect(inboxRows([lens("a", "support", [finding({})]), lens("b", "billing", [finding({})])])).toHaveLength(2);
  });

  it("takes the most urgent priority and latest sighting across merged findings", () => {
    const [row] = inboxRows([
      lens("a", "support", [finding({ priority: "low", last_seen: "2026-10-03T00:00:00Z" })]),
      lens("b", "support", [finding({ priority: "high", last_seen: "2026-10-01T00:00:00Z" })]),
    ]);
    expect(row.priority).toBe("high");
    expect(row.lastSeen).toBe("2026-10-03T00:00:00Z");
  });

  it("leaves out resolved, dismissed and pattern findings", () => {
    const rows = inboxRows([
      lens("a", "support", [
        finding({ id: "1", status: "resolved", title: "a" }),
        finding({ id: "2", status: "dismissed", title: "b" }),
        finding({ id: "3", kind: "pattern", title: "c" }),
      ]),
    ]);
    expect(rows).toEqual([]);
  });

  it("orders by priority, then by most recent", () => {
    const rows = inboxRows([
      lens("a", "x", [
        finding(ranked("1", "old high", "high", "2026-10-01T00:00:00Z")),
        finding(ranked("2", "new low", "low", "2026-10-03T00:00:00Z")),
        finding(ranked("3", "new high", "high", "2026-10-02T00:00:00Z")),
      ]),
    ]);
    expect(rows.map((r) => r.title)).toEqual(["new high", "old high", "new low"]);
  });
});

const step = (overrides: Partial<Step>): Step =>
  ({
    at: "2026-10-03T12:00:00Z",
    kind: "model",
    label: "Reviewed a run",
    model: "gpt-5.6",
    purpose: "extract",
    prompt_tokens: 0,
    completion_tokens: 0,
    cost: 0,
    ...overrides,
  }) as Step;

describe("filterInbox", () => {
  const rows = inboxRows([
    lens("a", "support", [finding({ id: "1", title: "one", priority: "high" })]),
    lens("b", "billing", [finding({ id: "2", title: "two", priority: "low" })]),
  ]);

  it("narrows to one agent and one priority, and lists every agent once", () => {
    expect(filterInbox(rows, { agent: "billing", priority: "all" }).map((r) => r.title)).toEqual(["two"]);
    expect(filterInbox(rows, { agent: ALL_AGENTS, priority: "high" }).map((r) => r.title)).toEqual(["one"]);
    expect(filterInbox(rows, { agent: "billing", priority: "high" })).toEqual([]);
    expect(inboxAgents(rows)).toEqual(["billing", "support"]);
  });
});

describe("step feed helpers", () => {
  it("summarizes a model call with its token count and cost", () => {
    expect(stepLine(step({ prompt_tokens: 1000, completion_tokens: 200, cost: 0.0031 }))).toBe(
      "Reviewed a run · 1.2k tok · $0.0031",
    );
    expect(stepLine(step({ kind: "stage", label: "Grouping observations" }))).toBe("Grouping observations");
  });

  it("lists models used, most used first, ignoring stage steps", () => {
    expect(
      modelsUsed([
        step({ model: "claude-sonnet-5-5" }),
        step({ model: "gpt-5.6" }),
        step({ model: "gpt-5.6" }),
        step({ kind: "stage", model: "" }),
      ]),
    ).toEqual(["gpt-5.6", "claude-sonnet-5-5"]);
  });
});

describe("scheduleLabel", () => {
  const now = Date.parse("2026-10-03T12:00:00Z");

  it("says how often it runs and when the next run is", () => {
    expect(scheduleLabel(lens("a", "x", []), now)).toBe("every 15m · next in 10m");
    expect(scheduleLabel(lens("a", "x", [], { settings: { interval_minutes: 120 } }), now)).toBe(
      "every 2h · next in 10m",
    );
  });

  it("calls out paused and overdue investigations", () => {
    expect(scheduleLabel(lens("a", "x", [], { settings: { enabled: false } }), now)).toBe("paused");
    expect(scheduleLabel(lens("a", "x", []), Date.parse("2026-10-03T12:30:00Z"))).toBe("every 15m · due now");
  });
});
