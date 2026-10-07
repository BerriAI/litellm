import { describe, expect, it } from "vitest";

import { configFrom, draftFrom, draftProblems, newRow, signalIds } from "./signalDraft";

const saved = {
  model: "jev",
  threshold: 0.5,
  signals: [{ id: "user_frustration", name: "User frustration", question: "Is the user frustrated?" }],
};

describe("signal drafts", () => {
  it("round-trips a saved config unchanged", () => {
    expect(configFrom(draftFrom(saved))).toEqual(saved);
  });

  it("keeps saved IDs on rename and derives unique IDs for new signals", () => {
    const rows = [
      { ...draftFrom(saved).rows[0], name: "Annoyed user" },
      { ...newRow("a"), name: "Repeated request!" },
      { ...newRow("b"), name: "Repeated request?" },
      { ...newRow("c"), name: "2nd try" },
    ];
    expect(signalIds(rows)).toEqual(["user_frustration", "repeated_request", "repeated_request_2", "signal_2nd_try"]);
  });

  it("reports blank, duplicate and out of range fields", () => {
    const draft = {
      model: "",
      thresholdPercent: 99,
      rows: [
        { ...newRow("a"), name: "Loop", question: "Does the agent loop?" },
        { ...newRow("b"), name: " loop ", question: "?" },
        newRow("c"),
      ],
    };
    const problems = draftProblems(draft);
    expect(problems.any).toBe(true);
    expect(problems.threshold).toBe("Use a whole number from 5 to 95");
    expect(problems.rows.get("a")).toBeUndefined();
    expect(problems.rows.get("b")).toEqual({
      name: "Another signal has this name",
      question: "Ask a yes or no question about the run",
    });
    expect(problems.rows.get("c")?.name).toBe("Name the signal");
  });
});
