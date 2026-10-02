import { describe, expect, it } from "vitest";
import { findRootBlocks } from "./rootBlocks";

describe("findRootBlocks", () => {
  it("maps each top-level request key to the lines its value spans", () => {
    const text = JSON.stringify(
      {
        model: "jev-latest",
        state: { message: "Hi" },
        questions: { escalate: { type: "noul", instructions: "Escalate?" } },
      },
      null,
      2,
    );

    expect(findRootBlocks(text)).toEqual([
      { key: "model", startLine: 1, endLine: 1 },
      { key: "state", startLine: 2, endLine: 4 },
      { key: "questions", startLine: 5, endLine: 10 },
    ]);
  });

  it("ignores nested keys, unrelated root keys and braces or quotes inside strings", () => {
    const text = [
      "{",
      '  "temperature": 0,',
      '  "state": "a } tricky \\" { string",',
      '  "questions": {',
      '    "model": { "state": "nested" }',
      "  }",
      "}",
    ].join("\n");

    expect(findRootBlocks(text)).toEqual([
      { key: "state", startLine: 2, endLine: 2 },
      { key: "questions", startLine: 3, endLine: 5 },
    ]);
  });

  it("finds every block on a single minified line", () => {
    expect(findRootBlocks('{"state":"Hi","questions":{"a":{"type":"noul"}}}')).toEqual([
      { key: "state", startLine: 0, endLine: 0 },
      { key: "questions", startLine: 0, endLine: 0 },
    ]);
  });

  it("keeps highlighting an unfinished block while the user is typing", () => {
    expect(findRootBlocks('{\n  "state": "Hi",\n  "questions": {\n    "a": {')).toEqual([
      { key: "state", startLine: 1, endLine: 1 },
      { key: "questions", startLine: 2, endLine: 3 },
    ]);
  });
});
