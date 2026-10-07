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
  const feedback = [
    entry("alice", 2, "2026-03-01T12:00:00Z"),
    entry("bob", 9, "2026-03-01T12:05:00Z"),
    entry("carol", 7, "2026-03-01T12:03:00Z"),
  ];

  it("separates the viewer's entry and lists everyone else newest first", () => {
    const view = feedbackView({ feedback, viewer: "alice" });
    expect(view.mine?.author).toBe("alice");
    expect(view.others.map((item) => item.author)).toEqual(["bob", "carol"]);
  });

  it("averages every score including the viewer's", () => {
    expect(feedbackView({ feedback, viewer: "alice" }).average).toBe(6);
  });

  it("never claims an entry for an anonymous viewer", () => {
    const view = feedbackView({ feedback: [entry("", 5, "2026-03-01T12:00:00Z")], viewer: "" });
    expect(view.mine).toBeNull();
    expect(view.others).toHaveLength(1);
  });

  it("has no average when nobody has rated the run", () => {
    expect(feedbackView({ feedback: [], viewer: "alice" })).toEqual({ mine: null, others: [], average: null });
  });
});
