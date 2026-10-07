import { describe, expect, it } from "vitest";

import { SIGNAL_LIBRARY } from "../../model/signals";
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
      { ...newRow("a"), name: "Repeat request!" },
      { ...newRow("b"), name: "Repeat request?" },
      { ...newRow("c"), name: "2nd try" },
    ];
    expect(signalIds(rows)).toEqual(["user_frustration", "repeat_request", "repeat_request_2", "signal_2nd_try"]);
  });

  it("does not assign a library ID to a new custom question", () => {
    const rows = [{ ...newRow("custom"), name: "Tool failure", question: "Does this custom signal apply?" }];

    expect(signalIds(rows)[0]).not.toBe("tool_failure");
  });

  it("keeps custom and library signals distinct when their names match", () => {
    const toolFailure = SIGNAL_LIBRARY.find((signal) => signal.id === "tool_failure");
    if (!toolFailure) throw new Error("Tool failure is missing from the signal library");
    const customQuestion = "Does this custom signal apply?";
    const draft = {
      model: "jev",
      thresholdPercent: 50,
      rows: [
        { ...newRow("library"), name: toolFailure.name, question: toolFailure.question },
        { ...newRow("custom"), id: toolFailure.id, name: toolFailure.name, question: customQuestion },
      ],
    };

    expect(configFrom(draft).signals).toEqual([
      { id: "tool_failure_2", name: "Tool failure", question: toolFailure.question },
      { id: "tool_failure", name: "Tool failure", question: customQuestion },
    ]);
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
