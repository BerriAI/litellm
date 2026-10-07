import { describe, expect, it } from "vitest";

import type { Feedback } from "../../types";
import { feedbackView } from "./feedback";

const entry = (author: string, score: number, updated_at: string): Feedback => ({
  author,
  score,
  comment: "",
  trace_id: "t1",
  trace_ref: "",
  created_at: updated_at,
  updated_at,
});

describe("feedbackView", () => {
  it("lists every end user's feedback newest first with the average and lowest score", () => {
    const view = feedbackView([
      entry("alice", 2, "2026-03-01T12:00:00Z"),
      entry("bob", 9, "2026-03-01T12:05:00Z"),
      entry("carol", 7, "2026-03-01T12:03:00Z"),
    ]);
    expect(view?.entries.map((item) => item.author)).toEqual(["bob", "carol", "alice"]);
    expect(view?.average).toBe(6);
    expect(view?.lowest).toBe(2);
  });

  it("has nothing to show when nobody rated the run", () => {
    expect(feedbackView([])).toBeNull();
  });
});
