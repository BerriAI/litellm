import { describe, expect, it } from "vitest";

import {
  UNKNOWN_AGENT,
  inboxRows,
  inboxFinding,
  filterInbox,
  findFinding,
  findingAgents,
  findingKey,
  openFindings,
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
    scope: { all_teams: true, team_id: "", api_key_hash: "" },
    settings: {
      name: id,
      agent_name: agent,
      service: "",
      enabled: true,
      interval_minutes: 15,
      checks: [{ id: "c", instruction: "Check tool errors" }],
      ...settings,
    },
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

describe("openFindings", () => {
  it("leaves out resolved, dismissed and pattern findings", () => {
    const owner = lens("a", "support", [
      finding({ id: "1", status: "resolved", title: "a" }),
      finding({ id: "2", status: "dismissed", title: "b" }),
      finding({ id: "3", kind: "pattern", title: "c" }),
      finding({ id: "4", title: "d" }),
    ]);
    expect(openFindings(owner).map((f) => f.title)).toEqual(["d"]);
  });

  it("orders by priority, then by most recent", () => {
    const owner = lens("a", "x", [
      finding(ranked("1", "old high", "high", "2026-10-01T00:00:00Z")),
      finding(ranked("2", "new low", "low", "2026-10-03T00:00:00Z")),
      finding(ranked("3", "new high", "high", "2026-10-02T00:00:00Z")),
    ]);
    expect(openFindings(owner).map((f) => f.title)).toEqual(["new high", "old high", "new low"]);
  });
});

describe("findFinding", () => {
  it("resolves a key to the investigation that owns the finding, even when ids repeat across investigations", () => {
    const a = lens("a", "support", [finding({ id: "same", title: "from a" })]);
    const b = lens("b", "support", [finding({ id: "same", title: "from b" })]);
    const found = findFinding([a, b], findingKey(b, b.findings[0]));
    expect(found?.lens.id).toBe("b");
    expect(found?.finding.title).toBe("from b");
    expect(findFinding([a, b], "missing:same")).toBeUndefined();
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

it("groups matching findings by agent, combines distinct runs and evidence, and takes the highest priority", () => {
  const quote = { execution_id: "run-1", span_id: "step", quote: "Failed", role: "support" as const };
  const firstInput: Partial<Finding> = {
    title: "Tool errors swallowed",
    occurrences: ["run-1"],
    evidence: [quote],
    priority: "low",
  };
  const first = finding(firstInput);
  const secondInput: Partial<Finding> = {
    title: " tool ERRORS swallowed ",
    occurrences: ["run-1", "run-2"],
    evidence: [quote],
    priority: "high",
    last_seen: "2026-10-02T00:00:00Z",
  };
  const second = finding(secondInput);
  const rows = inboxRows([
    lens("a", "support", [first]),
    lens("b", "support", [second]),
    lens("c", "billing", [first]),
  ]);
  expect(rows).toHaveLength(2);
  expect(rows[0]).toMatchObject({ priority: "high", runs: 2, lastSeen: second.last_seen });
  expect(rows[0].sources.map(({ lens }) => lens.id)).toEqual(["a", "b"]);
  expect(inboxFinding(rows[0])).toMatchObject({ occurrences: ["run-1", "run-2"], evidence: [quote], priority: "high" });
  expect(filterInbox(rows, { agent: "support", priority: "high" })).toEqual([rows[0]]);
  expect(filterInbox(rows, { agent: "billing", priority: "high" })).toEqual([]);
});

it("keeps only open issues in the cross-investigation inbox", () => {
  expect(
    inboxRows([
      lens("a", "support", [
        finding({ kind: "pattern" }),
        finding({ status: "resolved" }),
        finding({ status: "dismissed" }),
      ]),
    ]),
  ).toEqual([]);
});

it("keeps identical titles separate when their checks or visibility scopes differ", () => {
  const first = lens("a", "support", [finding({})]);
  const otherCheck = lens("b", "support", [finding({ check_id: "other" })]);
  const otherInstruction = lens("c", "support", [finding({})], {
    settings: { checks: [{ id: "c", instruction: "A different review criterion" }] },
  });
  const otherTeam = { ...first, id: "d", scope: { ...first.scope, team_id: "another-team" } };
  const rows = inboxRows([first, otherCheck, otherInstruction, otherTeam]);
  expect(rows).toHaveLength(4);
  expect(rows.map((row) => row.sources.map(({ lens }) => lens.id))).toEqual([["a"], ["b"], ["c"], ["d"]]);
});
