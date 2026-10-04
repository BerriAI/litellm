import { describe, expect, it } from "vitest";
import {
  analysisModel,
  conclusions,
  decidedReviews,
  focusedReview,
  grownGroups,
  inGroup,
  share,
  issueCount,
  shortVerdict,
  stripState,
  tickerLine,
  liveJob,
  liveStats,
  outcome,
  playbackPhase,
  playbackReducer,
  providerOf,
  queueRows,
  rateLabel,
  reviewKey,
  shownCount,
  startPlayback,
  stepDuration,
  tokenLabel,
  verdictLine,
} from "./live";
import type { Job, Review } from "./types";

function review(id: string, overrides: Partial<Review> = {}): Review {
  return {
    execution_id: id,
    trace_id: `trace-${id}`,
    agent: "support-bot",
    name: id,
    spans: [],
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

  it("names the agent and short trace id in the ticker, falling back to the run name", () => {
    expect(tickerLine(review("a", { trace_id: "a91f3c02deadbeef" }))).toBe("reading support-bot · a91f3c02");
    expect(tickerLine(review("a", { agent: "", name: "refund", trace_id: "7d21" }))).toBe("reading refund · 7d21");
  });

  it("leads the verdict line with the issue summary over patterns", () => {
    expect(verdictLine(review("a", { verdicts: [pattern("p", "fine"), issue("i", "made it up")] }))).toBe(
      "made it up",
    );
    expect(verdictLine(review("a"))).toBe("No issue observed");
    expect(verdictLine(review("a", { cannot_assess: true }))).toBe("Not enough evidence to judge");
  });
});

describe("conclusions", () => {
  it("counts traces per check and kind, ranking issues before patterns, then by count", () => {
    const reviews = [
      review("a", { verdicts: [pattern("calm"), issue("invented", "first")] }),
      review("b", { verdicts: [pattern("calm"), pattern("calm")] }),
      review("c", { verdicts: [issue("unhappy_user")] }),
      review("d", { verdicts: [issue("invented", "latest"), pattern("invented")] }),
    ];
    const result = conclusions(reviews, [{ id: "invented", instruction: "Invents answers", enabled: true }]);
    expect(result.map((c) => [c.checkId, c.count, c.issue])).toEqual([
      ["invented", 2, true],
      ["unhappy_user", 1, true],
      ["calm", 2, false],
      ["invented", 1, false],
    ]);
    expect(result[0].label).toBe("Invents answers");
    expect(result[0].latest).toBe("latest");
    expect(result[1].label).toBe("Unhappy user");
  });

  it("is empty when nothing was flagged", () => {
    expect(conclusions([review("a"), review("b")])).toEqual([]);
  });

  it("filters traces to a group and lets everything through without one", () => {
    const [invented] = conclusions([review("a", { verdicts: [issue("invented")] })]);
    expect(inGroup(review("a", { verdicts: [issue("invented")] }), invented.key)).toBe(true);
    expect(inGroup(review("b", { verdicts: [pattern("invented")] }), invented.key)).toBe(false);
    expect(inGroup(review("c"), null)).toBe(true);
  });

  it("reports which groups gained a trace so they can flash", () => {
    const before = conclusions([review("a", { verdicts: [issue("x"), pattern("y")] })]);
    const after = conclusions([
      review("a", { verdicts: [issue("x"), pattern("y")] }),
      review("b", { verdicts: [issue("x"), issue("z")] }),
    ]);
    expect([...grownGroups(before, after)].sort()).toEqual(["issue:x", "issue:z"]);
    expect(grownGroups(after, after).size).toBe(0);
  });

  it("gives a bar share bounded to the total", () => {
    expect(share(3, 12)).toBe(0.25);
    expect(share(5, 0)).toBe(0);
    expect(share(9, 4)).toBe(1);
  });

  it("only counts the trace being read once its verdict is on screen", () => {
    const [a, b] = [review("a"), review("b")];
    expect(decidedReviews({ played: [a], current: b }, false)).toEqual([a]);
    expect(decidedReviews({ played: [a], current: b }, true)).toEqual([a, b]);
  });
});

describe("playback pacing", () => {
  it("slows to a full window for one review and speeds up as the backlog grows", () => {
    expect(stepDuration(1)).toBe(2400);
    expect(stepDuration(0)).toBe(2400);
    expect(stepDuration(2)).toBeLessThan(stepDuration(1));
  });

  it("keeps every trace on screen long enough to read, however large the backlog", () => {
    expect(stepDuration(3)).toBeGreaterThanOrEqual(1200);
    expect(stepDuration(10_000)).toBeGreaterThanOrEqual(1200);
  });

  it("highlights spans one at a time, then types reasoning, then leaves the verdict up", () => {
    const duration = 2000;
    expect(playbackPhase(0, duration, 4, 100)).toEqual({ span: 0, typed: 0, verdict: false });
    expect(playbackPhase(duration * 0.29, duration, 4, 100).span).toBe(3);
    const typing = playbackPhase(duration * 0.475, duration, 4, 100);
    expect(typing).toEqual({ span: -1, typed: 50, verdict: false });
    const done = playbackPhase(duration * 0.65, duration, 4, 100);
    expect(done).toEqual({ span: -1, typed: 100, verdict: true });
    expect(playbackPhase(duration * 0.99, duration, 4, 100)).toEqual(done);
  });

  it("shows a settled review whole", () => {
    expect(playbackPhase(0, 0, 4, 100)).toEqual({ span: -1, typed: 100, verdict: true });
  });
});

describe("playback queue", () => {
  const reviews = ["a", "b", "c", "d", "e"].map((id) => review(id));

  it("opens a live job replaying the last few reviews and a finished job on its final review", () => {
    const live = startPlayback(reviews, true);
    expect(live.played.map((r) => r.execution_id)).toEqual(["a", "b"]);
    expect(live.current).toBeNull();
    expect(live.pending.map((r) => r.execution_id)).toEqual(["c", "d", "e"]);

    const done = startPlayback(reviews, false);
    expect(done.played.map((r) => r.execution_id)).toEqual(["a", "b", "c", "d"]);
    expect(done.current?.execution_id).toBe("e");
    expect(done.pending).toEqual([]);
  });

  it("enqueues only reviews it has not seen, so repeated polls do not replay", () => {
    const start = startPlayback(reviews.slice(0, 2), true);
    const polled = playbackReducer(start, { type: "enqueue", reviews: reviews.slice(0, 4) });
    expect(polled.pending.map((r) => r.execution_id)).toEqual(["a", "b", "c", "d"]);
    expect(playbackReducer(polled, { type: "enqueue", reviews: reviews.slice(0, 4) })).toBe(polled);
  });

  it("treats a re-reviewed execution at a new time as new", () => {
    const start = startPlayback([reviews[0]], false);
    const again = review("a", { at: "2026-10-03T17:00:00Z" });
    expect(reviewKey(again)).not.toBe(reviewKey(reviews[0]));
    expect(playbackReducer(start, { type: "enqueue", reviews: [again] }).pending).toEqual([again]);
  });

  it("advances one review per step and holds until the step finishes", () => {
    const start = startPlayback(reviews.slice(0, 2), true);
    expect(start.pending).toHaveLength(2);
    const first = playbackReducer(start, { type: "tick", now: 1000 });
    expect(first.current?.execution_id).toBe("a");
    expect(first.duration).toBe(stepDuration(2));
    expect(playbackReducer(first, { type: "tick", now: 1000 + first.duration - 1 })).toBe(first);
    const second = playbackReducer(first, { type: "tick", now: 1000 + first.duration });
    expect(second.current?.execution_id).toBe("b");
    expect(second.played.map((r) => r.execution_id)).toEqual(["a"]);
    expect(playbackReducer(second, { type: "tick", now: 1e9 })).toBe(second);
  });

  it("catches up on a big backlog by skipping to the newest few, never by shortening steps", () => {
    const many = Array.from({ length: 40 }, (_, n) => review(`r${n}`));
    const step = playbackReducer(startPlayback([], true), { type: "enqueue", reviews: many });
    const first = playbackReducer(step, { type: "tick", now: 0 });
    expect(first.current?.execution_id).toBe("r37");
    expect(first.pending.map((r) => r.execution_id)).toEqual(["r38", "r39"]);
    expect(first.played.map((r) => r.execution_id)).toEqual(many.slice(0, 37).map((r) => r.execution_id));
    expect(first.duration).toBeGreaterThanOrEqual(1200);
    expect(playbackReducer(first, { type: "tick", now: 1199 })).toBe(first);
  });

  it("replays a finished run, jumping to the last few", () => {
    const done = startPlayback(reviews, false);
    const again = playbackReducer(done, { type: "replay" });
    expect(again.current).toBeNull();
    expect(again.played).toEqual([]);
    expect(again.pending.map((r) => r.execution_id)).toEqual(["a", "b", "c", "d", "e"]);
    expect(playbackReducer(again, { type: "tick", now: 0 }).current?.execution_id).toBe("c");
  });

  it("settles everything at once for reduced motion", () => {
    const start = startPlayback(reviews, true);
    const settled = playbackReducer(start, { type: "settle" });
    expect(settled.current?.execution_id).toBe("e");
    expect(settled.played.map((r) => r.execution_id)).toEqual(["a", "b", "c", "d"]);
    expect(settled.pending).toEqual([]);
  });

  it("lists the newest review first, starting with the one being read", () => {
    const start = startPlayback(reviews, false);
    expect(queueRows(start, 3).map((r) => r.execution_id)).toEqual(["e", "d", "c"]);
    const live = playbackReducer(startPlayback(reviews, true), { type: "tick", now: 0 });
    expect(queueRows(live, 10).map((r) => r.execution_id)).toEqual(["c", "b", "a"]);
  });

  it("counts reviews beyond the capped list without counting the unplayed backlog", () => {
    const start = startPlayback(reviews, true);
    expect(shownCount(5, start)).toBe(2);
    expect(shownCount(120, start)).toBe(117);
  });
});

describe("which job the live run shows", () => {
  const job = (id: string, status: Job["status"], reviews: Review[]) => ({ id, status, reviews }) as unknown as Job;

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
    expect(stripState(base, MODEL)).toEqual({ kind: "waiting", message: "Reading 328 traces with cerebras/gpt-oss-120b…" });
    expect(stripState({ ...base, stage: "Grouping observations" }, "")).toEqual({
      kind: "waiting",
      message: "Grouping observations…",
    });
    expect(stripState({ ...base, status: "queued" }, MODEL).kind).toBe("waiting");
  });

  it("shows the job error plainly when the run failed", () => {
    expect(stripState({ ...base, status: "failed", error: "Anthropic: credit balance too low" }, MODEL)).toEqual({
      kind: "failed",
      message: "Anthropic: credit balance too low",
    });
    expect(stripState({ ...base, status: "failed" }, MODEL)).toEqual({ kind: "failed", message: "The investigation failed" });
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
    expect(issueCount({ status: "completed", findings, reviews: [issueReview], reviewed: 328 })).toEqual({
      count: 2,
      scope: "findings",
    });
  });

  it("labels counts honestly when only the last reviews are kept", () => {
    const reviews = [issueReview, review("y"), issueReview];
    expect(issueCount({ status: "running", findings: null, reviews, reviewed: 3 })).toEqual({ count: 2, scope: "" });
    expect(issueCount({ status: "running", findings: null, reviews, reviewed: 120 })).toEqual({
      count: 2,
      scope: "in last 3 reviewed",
    });
  });
});

describe("drawer focus", () => {
  const reviews = [review("a"), review("b")];

  it("follows the live review until one is picked", () => {
    expect(focusedReview(reviews, null, reviews[1])).toEqual({ review: reviews[1], following: true });
    expect(focusedReview(reviews, reviewKey(reviews[0]), reviews[1])).toEqual({ review: reviews[0], following: false });
  });

  it("goes back to live when the picked review was dropped by the cap", () => {
    expect(focusedReview(reviews, "gone@x", reviews[1])).toEqual({ review: reviews[1], following: true });
  });
});

describe("short verdict", () => {
  it("prefers the issue, otherwise says no issues or not enough evidence", () => {
    expect(shortVerdict(review("a", { verdicts: [pattern("p"), issue("i", "made it up")] }))).toBe("made it up");
    expect(shortVerdict(review("a", { verdicts: [pattern("p")] }))).toBe("no issues");
    expect(shortVerdict(review("a", { cannot_assess: true }))).toBe("not enough evidence");
  });
});

describe("live stats", () => {
  const job = {
    reviewed: 30,
    cost: 0.042,
    created_at: "2026-10-03T16:00:00Z",
    finished_at: null,
    steps: [
      { at: "2026-10-03T16:00:05Z", kind: "stage", label: "Reading executions", model: "", purpose: "", cost: 0,
        prompt_tokens: 0, completion_tokens: 0 },
      { at: "2026-10-03T16:00:06Z", kind: "model", label: "extract", model: "m", purpose: "extract", cost: 0.01,
        prompt_tokens: 1200, completion_tokens: 300 },
      { at: "2026-10-03T16:00:07Z", kind: "model", label: "extract", model: "m", purpose: "extract", cost: 0.01,
        prompt_tokens: 800, completion_tokens: 200 },
    ],
  } as unknown as Job;

  it("measures rate from when reading started and sums model tokens", () => {
    const stats = liveStats(job, Date.parse("2026-10-03T16:00:15Z"));
    expect(stats.elapsedSeconds).toBe(10);
    expect(stats.perSecond).toBe(3);
    expect(stats.tokens).toBe(2500);
    expect(stats.cost).toBe(0.042);
  });

  it("stops the clock when the job finishes", () => {
    const finished = { ...job, finished_at: "2026-10-03T16:00:25Z" };
    expect(liveStats(finished, Date.parse("2026-10-03T18:00:00Z")).elapsedSeconds).toBe(20);
  });

  it("has no rate before anything is reviewed", () => {
    expect(liveStats({ ...job, reviewed: 0 }, Date.parse("2026-10-03T16:00:15Z")).perSecond).toBeNull();
  });

  it("formats rate and tokens for the footer", () => {
    expect(rateLabel(null)).toBe("–");
    expect(rateLabel(2.25)).toBe("2.3 traces/s");
    expect(rateLabel(0.5)).toBe("30.0 traces/min");
    expect(tokenLabel(950)).toBe("950 tok");
    expect(tokenLabel(2500)).toBe("2.5k tok");
    expect(tokenLabel(3_400_000)).toBe("3.4M tok");
  });
});
