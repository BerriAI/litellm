import { describe, expect, it } from "vitest";

import {
  holdMs,
  laneText,
  releasedReviews,
  stageTick,
  startStage,
  stepStage,
  typedChars,
  type StageInput,
} from "./stage";
import type { Review } from "./types";

function review(id: string, reasoning = "Checked the tool output. It matched."): Review {
  return {
    execution_id: id,
    trace_id: `trace-${id}`,
    agent: "support-bot",
    name: id,
    spans: [],
    tool_calls: [],
    reasoning,
    verdicts: [],
    cannot_assess: false,
    model: "cerebras/gpt-oss-120b",
    duration_ms: 900,
    at: "2026-10-03T16:00:00Z",
  };
}

const reading = (id: string) => ({
  execution_id: id,
  trace_id: `trace-${id}`,
  agent: "bot",
  started_at: "2026-10-03T16:00:00Z",
});
const input = (overrides: Partial<StageInput>): StageInput => ({
  reading: [],
  reviews: [],
  now: 0,
  slots: 4,
  running: true,
  charMs: 3,
  ...overrides,
});
const keys = (stage: ReturnType<typeof startStage>) => stage.lanes.map((lane) => lane.key);

describe("lane text", () => {
  it("keeps the first two sentences and caps the length", () => {
    expect(laneText("One. Two. Three.")).toBe("One. Two.");
    const long = laneText(`${"word ".repeat(80)}end.`);
    expect(long.length).toBeLessThanOrEqual(220);
    expect(long.endsWith("…")).toBe(true);
  });

  it("types at the given speed and holds the verdict after typing", () => {
    expect(typedChars({ landedAt: 1000 }, 1030, 3)).toBe(10);
    expect(typedChars({ landedAt: null }, 1030, 3)).toBe(0);
    expect(typedChars({ landedAt: 1000 }, 1000, 0)).toBe(Number.MAX_SAFE_INTEGER);
    expect(holdMs(review("a", "Short."), 3)).toBe(6 * 3 + 1200);
  });
});

describe("now reading stage", () => {
  it("does not replay reviews that already existed when the view opened", () => {
    const stage = startStage([review("old")]);
    const next = stepStage(stage, input({ reviews: [review("old")] }));
    expect(next).toBe(stage);
    expect(releasedReviews([review("old")], next).map((r) => r.execution_id)).toEqual(["old"]);
  });

  it("shows real in-flight traces, lands the review in the same lane, then hands it to the list", () => {
    const opened = stepStage(startStage([]), input({ reading: [reading("x"), reading("y")] }));
    expect(keys(opened)).toEqual(["x", "y"]);
    expect(opened.lanes.every((lane) => lane.review === null)).toBe(true);

    const landed = stepStage(opened, input({ reading: [reading("y")], reviews: [review("x")], now: 500 }));
    expect(keys(landed)).toEqual(["x", "y"]);
    expect(landed.lanes[0]).toMatchObject({ landedAt: 500 });
    expect(landed.released.has("x")).toBe(false);

    const hold = holdMs(review("x"), 3);
    const still = stepStage(landed, input({ reading: [reading("y")], reviews: [review("x")], now: 500 + hold - 1 }));
    expect(keys(still)).toEqual(["x", "y"]);
    const handed = stepStage(still, input({ reading: [reading("y")], reviews: [review("x")], now: 500 + hold }));
    expect(keys(handed)).toEqual(["y"]);
    expect(handed.released.has("x")).toBe(true);
  });

  it("gives newly arrived reviews a lane when the worker does not report what it is reading", () => {
    const stage = stepStage(startStage([]), input({ reviews: [review("a"), review("b")], now: 10 }));
    expect(keys(stage)).toEqual(["a", "b"]);
    expect(stage.lanes.every((lane) => lane.landedAt === 10)).toBe(true);
  });

  it("never lets a burst back up: extra reviews go straight to the list, oldest first", () => {
    const burst = ["a", "b", "c", "d", "e", "f"].map((id) => review(id));
    const stage = stepStage(startStage([]), input({ reviews: burst, slots: 2 }));
    expect(keys(stage)).toEqual(["e", "f"]);
    expect([...stage.released].sort()).toEqual(["a", "b", "c", "d"]);
  });

  it("drops a lane whose trace vanished without a review after a grace period", () => {
    const opened = stepStage(startStage([]), input({ reading: [reading("x")] }));
    const gone = stepStage(opened, input({ now: 1000 }));
    expect(keys(gone)).toEqual(["x"]);
    expect(keys(stepStage(gone, input({ now: 5000 })))).toEqual([]);
    expect(stepStage(gone, input({ now: 5000 })).released.has("x")).toBe(false);
  });

  it("releases everything once the run stops", () => {
    const opened = stepStage(startStage([]), input({ reading: [reading("x")], reviews: [review("a")] }));
    const stopped = stepStage(opened, input({ running: false, reviews: [review("a"), review("x")] }));
    expect(stopped.lanes).toEqual([]);
    expect([...stopped.released].sort()).toEqual(["a", "x"]);
  });
});

describe("stage tick", () => {
  it("ticks fast only while reasoning is typing, slowly for live timers, and stops when idle", () => {
    const landed = stepStage(startStage([]), input({ reviews: [review("a")], now: 0, charMs: 3 }));
    expect(stageTick(landed, true, 1, 3)).toBe(30);
    expect(stageTick(landed, true, 10_000, 3)).toBe(500);
    expect(stageTick(startStage([]), true, 0, 3)).toBe(500);
    expect(stageTick(startStage([]), false, 0, 3)).toBeNull();
  });
});
