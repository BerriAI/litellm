import { describe, expect, it } from "vitest";
import {
  analysisModel,
  conclusions,
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

  it("leads the verdict line with the issue summary over patterns", () => {
    expect(verdictLine(review("a", { verdicts: [pattern("p", "fine"), issue("i", "made it up")] }))).toBe(
      "made it up",
    );
    expect(verdictLine(review("a"))).toBe("No issue observed");
    expect(verdictLine(review("a", { cannot_assess: true }))).toBe("Not enough evidence to judge");
  });
});

describe("conclusions", () => {
  it("groups verdicts by check, counts them and ranks issues before patterns, then by count", () => {
    const reviews = [
      review("a", { verdicts: [pattern("calm"), issue("invented", "first")] }),
      review("b", { verdicts: [pattern("calm"), pattern("calm")] }),
      review("c", { verdicts: [issue("unhappy")] }),
      review("d", { verdicts: [issue("invented", "latest")] }),
    ];
    const result = conclusions(reviews, [{ id: "invented", instruction: "Invents answers", enabled: true }]);
    expect(result.map((c) => [c.checkId, c.count, c.issue])).toEqual([
      ["invented", 2, true],
      ["unhappy", 1, true],
      ["calm", 3, false],
    ]);
    expect(result[0].label).toBe("Invents answers");
    expect(result[0].latest).toBe("latest");
    expect(result[1].label).toBe("unhappy");
  });

  it("is empty when nothing was flagged", () => {
    expect(conclusions([review("a"), review("b")])).toEqual([]);
  });
});

describe("playback pacing", () => {
  it("slows to a full window for one review and speeds up as the backlog grows", () => {
    expect(stepDuration(1)).toBe(2400);
    expect(stepDuration(2)).toBe(1200);
    expect(stepDuration(10)).toBeLessThan(stepDuration(2));
    expect(stepDuration(0)).toBe(2400);
  });

  it("streams a large backlog at 150ms per review or faster", () => {
    expect(stepDuration(16)).toBeLessThanOrEqual(150);
    expect(stepDuration(60)).toBeLessThanOrEqual(150);
    expect(stepDuration(10_000)).toBe(60);
  });

  it("highlights spans one at a time, then types reasoning, then shows the verdict", () => {
    const duration = 2000;
    expect(playbackPhase(0, duration, 4, 100)).toEqual({ span: 0, typed: 0, verdict: false });
    expect(playbackPhase(duration * 0.3, duration, 4, 100).span).toBe(3);
    const typing = playbackPhase(duration * 0.575, duration, 4, 100);
    expect(typing).toEqual({ span: -1, typed: 50, verdict: false });
    expect(playbackPhase(duration * 0.85, duration, 4, 100)).toEqual({ span: -1, typed: 100, verdict: true });
  });

  it("skips the animation entirely when the backlog forces short steps", () => {
    expect(playbackPhase(0, 300, 4, 100)).toEqual({ span: -1, typed: 100, verdict: true });
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
