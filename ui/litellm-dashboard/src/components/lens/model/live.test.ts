import { describe, expect, it } from "vitest";
import {
  analysisModel,
  briefReasoning,
  checkLabel,
  conclusions,
  doneLine,
  durationLabel,
  newestFirst,
  nowLine,
  inFlight,
  grownGroups,
  inGroup,
  share,
  issueCount,
  appendPage,
  EMPTY_FEED,
  shortVerdict,
  stripState,
  liveJob,
  outcome,
  providerOf,
  polling,
  activeActivities,
  activityOperation,
  activityPhase,
  toolCallSummary,
  reviewScope,
} from "./live";
import type { Activity, Job, Review } from "./types";

function review(id: string, overrides: Partial<Review> = {}): Review {
  return {
    execution_id: id,
    trace_id: `trace-${id}`,
    agent: "support-bot",
    name: id,
    spans: [],
    tool_calls: [],
    reasoning: "",
    verdicts: [],
    cannot_assess: false,
    model: "cerebras/gpt-oss-120b",
    duration_ms: 100,
    at: "2026-10-03T16:00:00Z",
    ...overrides,
  };
}

const issue = (check: string, summary = check) => ({ check_id: check, kind: "issue" as const, summary });
const pattern = (check: string, summary = check) => ({ check_id: check, kind: "pattern" as const, summary });

describe("provider from model", () => {
  it("takes the lowercase prefix before the first slash", () => {
    expect(providerOf("Cerebras/gpt-oss-120b")).toBe("cerebras");
    expect(providerOf("openrouter/meta/llama")).toBe("openrouter");
  });

  it("has no provider for a bare model alias", () => {
    expect(providerOf("analysis")).toBe("");
    expect(providerOf("/weird")).toBe("");
  });

  it("resolves a bare model name through the model catalog", () => {
    const catalog = {
      "gpt-5.6": { litellm_provider: "openai" },
      "cerebras/gpt-oss-120b": { litellm_provider: "cerebras" },
    };
    expect(providerOf("gpt-5.6", catalog)).toBe("openai");
    expect(providerOf("my-alias", catalog)).toBe("");
    expect(analysisModel(["", "gpt-5.6"], catalog)).toBe("gpt-5.6");
  });

  it("prefers a model that names its provider over a bare alias", () => {
    expect(analysisModel(["analysis", "", "cerebras/gpt-oss-120b"])).toBe("cerebras/gpt-oss-120b");
    expect(analysisModel(["", "analysis"])).toBe("analysis");
    expect(analysisModel([])).toBe("");
  });
});

describe("review outcome", () => {
  it("is an issue when any verdict is an issue", () => {
    expect(outcome(review("a", { verdicts: [pattern("p"), issue("i")] }))).toBe("issue");
  });

  it("is clear when only patterns are seen", () => {
    expect(outcome(review("a", { verdicts: [pattern("p")] }))).toBe("clear");
  });

  it("is unknown when the model cannot assess, even with an issue", () => {
    expect(outcome(review("a", { cannot_assess: true, verdicts: [issue("i")] }))).toBe("unknown");
  });

  it("keeps the first few sentences of the reasoning", () => {
    expect(briefReasoning("One. Two? Three! Four. Five.")).toBe("One. Two? Three!");
    expect(briefReasoning("Saw 1.5 percent drop. Fine")).toBe("Saw 1.5 percent drop. Fine");
    expect(briefReasoning("  ")).toBe("");
  });
});

describe("conclusions", () => {
  it("makes one group per check, counting issue traces and noting pattern traces", () => {
    const reviews = [
      review("a", { verdicts: [pattern("calm"), issue("invented", "first")] }),
      review("b", { verdicts: [pattern("calm"), pattern("calm")] }),
      review("c", { verdicts: [issue("unhappy_user")] }),
      review("d", { verdicts: [issue("invented", "latest"), pattern("invented", "fine")] }),
      review("e", { verdicts: [pattern("invented", "fine")] }),
    ];
    const result = conclusions(reviews, [{ id: "invented", instruction: "Invents answers", enabled: true }]);
    expect(result.map((c) => [c.checkId, c.count, c.noted, c.issue])).toEqual([
      ["invented", 2, 1, true],
      ["unhappy_user", 1, 0, true],
      ["calm", 0, 2, false],
    ]);
    expect(new Set(result.map((c) => c.key)).size).toBe(result.length);
    expect(result[0].label).toBe("Invents answers");
    expect(result[0].latest).toBe("latest");
    expect(result[1].label).toBe("Unhappy user");
  });

  it("labels a check by its humanized id when the instruction is long", () => {
    const long = "Agent takes a risky action (refund over limit, prod deploy) without required approval";
    expect(checkLabel("no_approval", long)).toBe("No approval");
    expect(checkLabel("expected_behavior", undefined)).toBe("Expected behavior");
    expect(checkLabel("x", "Invents answers")).toBe("Invents answers");
  });

  it("is empty when nothing was flagged", () => {
    expect(conclusions([review("a"), review("b")])).toEqual([]);
  });

  it("filters traces to a group and lets everything through without one", () => {
    const [invented] = conclusions([review("a", { verdicts: [issue("invented")] })]);
    expect(inGroup(review("a", { verdicts: [issue("invented")] }), invented.key)).toBe(true);
    expect(inGroup(review("b", { verdicts: [pattern("other")] }), invented.key)).toBe(false);
    expect(inGroup(review("c"), null)).toBe(true);
  });

  it("reports which groups gained a trace so they can flash", () => {
    const before = conclusions([review("a", { verdicts: [issue("x"), pattern("y")] })]);
    const after = conclusions([
      review("a", { verdicts: [issue("x"), pattern("y")] }),
      review("b", { verdicts: [issue("x"), issue("z")] }),
    ]);
    expect([...grownGroups(before, after)].sort()).toEqual(["x", "z"]);
    expect(grownGroups(after, after).size).toBe(0);
  });

  it("flashes a group that only gained a pattern trace", () => {
    const before = conclusions([review("a", { verdicts: [pattern("y")] })]);
    const after = conclusions([review("a", { verdicts: [pattern("y")] }), review("b", { verdicts: [pattern("y")] })]);
    expect([...grownGroups(before, after)]).toEqual(["y"]);
  });

  it("gives a bar share bounded to the total", () => {
    expect(share(3, 12)).toBe(0.25);
    expect(share(5, 0)).toBe(0);
    expect(share(9, 4)).toBe(1);
  });
});

describe("which job the live run shows", () => {
  const job = (id: string, status: Job["status"], reviews: Review[]) =>
    ({ id, status, reviews: [], reviewed: reviews.length }) as unknown as Job;

  it("shows the active job once it has reviews", () => {
    expect(liveJob([job("new", "running", [review("a")]), job("old", "completed", [review("b")])])?.id).toBe("new");
  });

  it("shows an active job before its first review so the wait is explained", () => {
    expect(liveJob([job("new", "queued", []), job("old", "completed", [review("b")])])?.id).toBe("new");
  });

  it("keeps the latest finished job viewable only when it has reviews", () => {
    expect(liveJob([job("done", "completed", [review("a")])])?.id).toBe("done");
    expect(liveJob([job("done", "completed", []), job("older", "completed", [review("b")])])).toBeUndefined();
    expect(liveJob([])).toBeUndefined();
  });
});

describe("strip state", () => {
  const base = {
    status: "running" as Job["status"],
    error: "",
    stage: "Reading executions",
    steps: [] as Job["steps"],
    coverage: { selected: 328 } as Job["coverage"],
    reviews: [] as Review[],
  };
  const MODEL = "cerebras/gpt-oss-120b";

  it("says what is happening before the first review instead of a silent spinner", () => {
    expect(stripState(base, MODEL)).toEqual({
      kind: "waiting",
      message: "Reading 328 traces with cerebras/gpt-oss-120b…",
    });
    expect(stripState({ ...base, stage: "Grouping observations" }, "")).toEqual({
      kind: "waiting",
      message: "Grouping observations…",
    });
    expect(stripState({ ...base, status: "queued" }, MODEL, "No worker connected.")).toEqual({
      kind: "waiting",
      message: "No worker connected.",
    });
  });

  it("shows the job error plainly when the run failed", () => {
    expect(stripState({ ...base, status: "failed", error: "Anthropic: credit balance too low" }, MODEL)).toEqual({
      kind: "failed",
      message: "Anthropic: credit balance too low",
    });
    expect(stripState({ ...base, status: "failed" }, MODEL)).toEqual({
      kind: "failed",
      message: "The investigation failed",
    });
  });

  it("surfaces a model error step while still running with nothing reviewed", () => {
    const steps = [{ kind: "error", label: "Model call failed: 400 credit balance too low" }] as Job["steps"];
    expect(stripState({ ...base, steps }, MODEL)).toEqual({
      kind: "failed",
      message: "Model call failed: 400 credit balance too low",
    });
    expect(stripState({ ...base, steps, reviews: [review("a")] }, MODEL).kind).toBe("reviewing");
  });

  it("is done for finished runs and reviewing once reviews arrive", () => {
    expect(stripState({ ...base, status: "completed" }, MODEL).kind).toBe("done");
    expect(stripState({ ...base, reviews: [review("a")] }, MODEL).kind).toBe("reviewing");
  });
});

describe("issue count", () => {
  const issueReview = review("x", { verdicts: [issue("i")] });

  it("uses the job's issue findings once the run completes", () => {
    const findings = [{ kind: "issue" }, { kind: "pattern" }, { kind: "issue" }] as Job["findings"];
    const completed = { status: "completed" as const, findings, reviews: [issueReview], reviewed: 328 };
    expect(issueCount(completed)).toEqual({ count: 2, scope: "findings" });
  });

  it("labels counts honestly when only the last reviews are kept", () => {
    const reviews = [issueReview, review("y"), issueReview];
    const running = { status: "running" as const, findings: null, reviews };
    expect(issueCount({ ...running, reviewed: 3 })).toEqual({ count: 2, scope: "" });
    expect(issueCount({ ...running, reviewed: 120 })).toEqual({
      count: 2,
      scope: "in 3 displayed reviews",
    });
  });
});

describe("incremental reviews", () => {
  const at = (id: string, minute: number) => review(id, { at: `2026-10-03T16:${String(minute).padStart(2, "0")}:00Z` });

  it("appends pages in arrival order, not by time, and advances the cursor to the reviewed count", () => {
    const first = appendPage(EMPTY_FEED, { reviews: [at("a", 5), at("b", 2)], reviewed: 2 });
    expect(first).toEqual({ reviews: [at("a", 5), at("b", 2)], cursor: 2 });
    const second = appendPage(first, { reviews: [at("c", 1)], reviewed: 3 });
    expect(second.reviews.map((r) => r.execution_id)).toEqual(["a", "b", "c"]);
    expect(second.cursor).toBe(3);
  });

  it("keeps the same feed for an empty page and never duplicates a re-sent review", () => {
    const feed = appendPage(EMPTY_FEED, { reviews: [at("a", 1)], reviewed: 1 });
    expect(appendPage(feed, { reviews: [], reviewed: 1 })).toBe(feed);
    expect(appendPage(feed, { reviews: [at("a", 1)], reviewed: 1 })).toBe(feed);
  });

  it("caps how many reviews are kept, dropping the oldest", () => {
    const many = Array.from({ length: 250 }, (_, n) => review(`r${n}`));
    const feed = appendPage(EMPTY_FEED, { reviews: many, reviewed: 250 });
    expect(feed.reviews).toHaveLength(200);
    expect(feed.reviews[0].execution_id).toBe("r50");
    expect(feed.cursor).toBe(250);
  });

  it("keeps polling a finished run until every review it reported has arrived", () => {
    const fetched = { reviews: [], cursor: 28 };
    expect(polling({ status: "running", reviewed: 28 }, fetched)).toBe(true);
    expect(polling({ status: "completed", reviewed: 30 }, fetched)).toBe(true);
    expect(polling({ status: "completed", reviewed: 30 }, { ...fetched, cursor: 30 })).toBe(false);
  });
});

describe("short verdict", () => {
  it("prefers the issue, otherwise says no issues or not enough evidence", () => {
    expect(shortVerdict(review("a", { verdicts: [pattern("p"), issue("i", "made it up")] }))).toBe("made it up");
    expect(shortVerdict(review("a", { verdicts: [pattern("p")] }))).toBe("no issues");
    expect(shortVerdict(review("a", { cannot_assess: true }))).toBe("not enough evidence");
  });
});

describe("honest live list", () => {
  it("lists completed reviews newest first in the order they finished, never re-sorted by time", () => {
    const reviews = [
      review("a", { at: "2026-10-03T16:05:00Z" }),
      review("b", { at: "2026-10-03T16:01:00Z" }),
      review("c"),
    ];
    expect(newestFirst(reviews, 10).map((r) => r.execution_id)).toEqual(["c", "b", "a"]);
    expect(newestFirst(reviews, 2).map((r) => r.execution_id)).toEqual(["c", "b"]);
    expect(reviews.map((r) => r.execution_id)).toEqual(["a", "b", "c"]);
  });

  it("says how many traces are in flight and how many are done", () => {
    const job = (reviewed: number, selected: number) => ({ reviewed, coverage: { selected } }) as unknown as Job;
    expect(nowLine(job(18, 30), 4)).toBe("18 of 30 · 4 in flight");
    expect(nowLine(job(30, 30), 0)).toBe("30 of 30");
    expect(nowLine(job(3, 0), 1)).toBe("3 done · 1 in flight");
  });

  it("only shows in-flight traces while the job runs", () => {
    const item = { execution_id: "x", trace_id: "t", agent: "bot", started_at: "2026-10-03T16:00:00Z" };
    const job = (status: Job["status"]) => ({ status, reading: [item] }) as unknown as Job;
    expect(inFlight(job("running"))).toEqual([item]);
    expect(inFlight(job("completed"))).toEqual([]);
    expect(inFlight({ status: "running" } as Job)).toEqual([]);
  });

  it("sums up a finished run from when reading started", () => {
    const job = {
      reviewed: 30,
      created_at: "2026-10-03T16:00:00Z",
      finished_at: "2026-10-03T16:00:41Z",
      steps: [{ kind: "stage", label: "Reading executions", at: "2026-10-03T16:00:10Z" }],
    } as unknown as Job;
    expect(doneLine(job)).toBe("Reviewed 30 traces in 31s with");
    expect(doneLine({ ...job, reviewed: 1, finished_at: null })).toBe("Reviewed 1 trace with");
  });

  it("formats review time", () => {
    expect(durationLabel(420)).toBe("420ms");
    expect(durationLabel(1420)).toBe("1.4s");
    expect(durationLabel(83_000)).toBe("1m 23s");
  });
});

describe("analysis activity", () => {
  const activity: Activity = {
    id: "candidate-1",
    phase: "investigate",
    label: "Check repeated grep failures",
    execution_ids: ["a", "b"],
    started_at: "2026-10-03T16:00:00Z",
    operations: ["python", "read"],
    tool_calls: [{ name: "search", calls: 2 }],
    finished: false,
  };

  it("uses actual phase and operations instead of calling candidate work a trace read", () => {
    expect(activityPhase(activity)).toBe("Investigating candidate");
    expect(activityOperation(activity)).toBe("Running Python · Reading trace content");
    expect(activityOperation({ phase: "group", operations: [] })).toBe("Grouping observations");
    expect(activityPhase({ phase: "reconcile" })).toBe("Combining candidate groups");
    expect(activityPhase({ phase: "load" })).toBe("Loading trace evidence");
    expect(activityPhase({ phase: "review" })).toBe("Reviewing trace");
    expect(activityOperation({ phase: "review", operations: ["checkpoint", "model"] })).toBe(
      "Compacting context · Analyzing evidence",
    );
  });

  it("keeps fast tool calls visible separately from the current operation", () => {
    expect(activityOperation({ ...activity, operations: ["model"] })).toBe("Analyzing evidence");
    expect(
      toolCallSummary([
        { name: "read", calls: 2 },
        { name: "python", calls: 1 },
        { name: "search", calls: 0 },
      ]),
    ).toBe("Read × 2 · Python × 1");
    expect(toolCallSummary()).toBe("");
    expect(toolCallSummary([{ name: "checkpoint", calls: 1 }])).toBe("Context compaction × 1");
  });

  it("clears active work after termination and ignores completed snapshot entries", () => {
    expect(activeActivities({ status: "running", activities: [activity, { ...activity, finished: true }] })).toEqual([
      activity,
    ]);
    expect(activeActivities({ status: "completed", activities: [activity] })).toEqual([]);
    expect(activeActivities({ status: "failed", activities: [activity] })).toEqual([]);
  });

  it("describes retained reviews without claiming a gap-free latest window", () => {
    expect(reviewScope(2048, 120)).toBe("120 displayed of 2048 reviewed");
    expect(reviewScope(20, 20)).toBe("");
  });

  it("still reports grouping and investigation stages after initial reviews arrive", () => {
    const job = {
      status: "running" as const,
      error: "",
      steps: [],
      coverage: { selected: 2048 } as Job["coverage"],
      reviews: [review("a")],
      stage: "Grouping observations",
    };
    expect(stripState(job, "analysis")).toEqual({ kind: "reviewing", message: "Grouping observations" });
    expect(stripState({ ...job, stage: "Checking original evidence" }, "analysis")).toEqual({
      kind: "reviewing",
      message: "Checking original evidence",
    });
  });
});
