import { describe, expect, it } from "vitest";
import { SYSTEM_ONE_EXAMPLE } from "./example";
import {
  blankQuestion,
  freeName,
  readForm,
  removeAt,
  renameKey,
  replaceAt,
  retype,
  withNoulCriterion,
  withoutKey,
  type NoulQuestion,
} from "./formPayload";

describe("readForm", () => {
  it("reads the example request as it is", () => {
    expect(readForm(JSON.stringify(SYSTEM_ONE_EXAMPLE))).toEqual(SYSTEM_ONE_EXAMPLE);
  });

  it("keeps fields the form does not edit, in the order the request had them", () => {
    const raw = JSON.stringify({
      metadata: { k: 1 },
      questions: { q: { note: "x", type: "noul", instructions: "i" } },
      state: "s",
    });
    const form = readForm(raw);
    expect(JSON.stringify(form)).toBe(raw);
  });

  it("reads a request with no state or questions yet", () => {
    expect(readForm("{}")).toEqual({});
  });

  it.each([
    ["broken JSON", "{"],
    ["a non-object", "[1]"],
    ["a non-text state", JSON.stringify({ state: { ticket: 1 } })],
    ["an unknown answer type", JSON.stringify({ questions: { q: { type: "rank", criteria: [] } } })],
    ["a non-text option description", JSON.stringify({ questions: { q: { type: "choice", criteria: { a: 1 } } } })],
    ["a non-text score level", JSON.stringify({ questions: { q: { type: "score", criteria: [1, 2] } } })],
    ["a non-text yes/no criterion", JSON.stringify({ questions: { q: { type: "noul", criteria: { true: 1 } } } })],
    ["non-text instructions", JSON.stringify({ questions: { q: { type: "noul", instructions: { a: 1 } } } })],
  ])("cannot show %s", (_, raw) => {
    expect(readForm(raw)).toBeUndefined();
  });
});

describe("freeName", () => {
  it("picks the first unused numbered name", () => {
    expect(freeName([], "question")).toBe("question_1");
    expect(freeName(["question_1", "question_3"], "question")).toBe("question_2");
    expect(freeName(["option_1", "option_2"], "option")).toBe("option_3");
  });
});

describe("record and list edits", () => {
  it("renames a key in place", () => {
    const renamed = renameKey({ a: 1, b: 2, c: 3 }, "b", "z");
    expect(Object.entries(renamed)).toEqual([
      ["a", 1],
      ["z", 2],
      ["c", 3],
    ]);
  });

  it("removes only the named key", () => {
    expect(withoutKey({ a: 1, b: 2 }, "a")).toEqual({ b: 2 });
  });

  it("replaces and removes list items by position", () => {
    expect(replaceAt(["a", "b", "c"], 1, "x")).toEqual(["a", "x", "c"]);
    expect(removeAt(["a", "b", "c"], 1)).toEqual(["a", "c"]);
  });
});

describe("retype", () => {
  const noul = { type: "noul", instructions: "Is it urgent?", note: "kept", criteria: { true: "Yes" } } as const;

  it("keeps instructions and other fields and starts the new type with blank criteria", () => {
    const asChoice = {
      type: "choice",
      instructions: "Is it urgent?",
      note: "kept",
      criteria: { option_1: "", option_2: "" },
    };
    const asScore = { type: "score", instructions: "Is it urgent?", note: "kept", criteria: ["", ""] };
    expect(retype(noul, "choice")).toEqual(asChoice);
    expect(retype(noul, "score")).toEqual(asScore);
    expect(retype(blankQuestion(), "noul")).toEqual({ type: "noul", instructions: "" });
  });

  it("keeps type as the first field so the JSON view reads the same as a new question", () => {
    expect(Object.keys(retype(noul, "choice"))).toEqual(["type", "instructions", "note", "criteria"]);
    expect(Object.keys(retype(noul, "score"))).toEqual(["type", "instructions", "note", "criteria"]);
    expect(Object.keys(retype(blankQuestion(), "noul"))).toEqual(["type", "instructions"]);
  });

  it("leaves a question alone when the type does not change", () => {
    expect(retype(noul, "noul")).toBe(noul);
  });
});

describe("withNoulCriterion", () => {
  const question: NoulQuestion = { type: "noul", instructions: "i", criteria: { true: "Yes", false: "No" } };

  it("sets one side and keeps the other", () => {
    expect(withNoulCriterion(question, "false", "Nope")).toEqual({
      type: "noul",
      instructions: "i",
      criteria: { true: "Yes", false: "Nope" },
    });
  });

  it("drops a side left blank and drops the criteria when both are blank", () => {
    const yesOnly = withNoulCriterion(question, "false", "");
    expect(yesOnly).toEqual({ type: "noul", instructions: "i", criteria: { true: "Yes" } });
    expect(withNoulCriterion(yesOnly, "true", "")).toEqual({ type: "noul", instructions: "i" });
  });

  it("adds criteria to a question that had none", () => {
    expect(withNoulCriterion({ type: "noul" }, "true", "Yes")).toEqual({ type: "noul", criteria: { true: "Yes" } });
  });
});
