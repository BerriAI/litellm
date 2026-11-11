import { describe, expect, it } from "vitest";
import { openAIDecisionsExample } from "./example";
import { readDecisionsForm, retypeDecisionQuestion, type DecisionsFormQuestion } from "./openAIDecisionsForm";
import { validateOpenAIDecisionsPayload } from "./validatePayload";

describe("readDecisionsForm", () => {
  it("round-trips examples and preserves input order, extension fields, false choices and null names", () => {
    const example = openAIDecisionsExample();
    expect(readDecisionsForm(JSON.stringify(example))).toEqual(JSON.parse(JSON.stringify(example)));
    const payload = {
      metadata: { source: "playground" },
      input: [{ role: "user", content: [{ type: "input_image", image_url: "https://example.com/image.png" }] }],
      safety_identifier: null,
      questions: [{ type: "choice", name: null, instructions: "Choose", choices: [{ value: false }, { value: true }] }],
    };
    expect(JSON.stringify(readDecisionsForm(JSON.stringify(payload)))).toEqual(JSON.stringify(payload));
  });

  it.each([
    {},
    { input: "", questions: [] },
    { input: "text", questions: [{ type: "choice", choices: [] }] },
    { input: "text", questions: [{ type: "score", instructions: "Score", levels: [{ label: "0" }] }] },
    {
      input: "text",
      questions: [{ type: "choice", instructions: "Choose", choices: [{ value: "same" }, { value: "same" }] }],
    },
  ])("keeps incomplete requests editable without marking them valid for Send: %j", (payload) => {
    const raw = JSON.stringify(payload);
    expect(readDecisionsForm(raw)).toEqual(payload);
    expect(validateOpenAIDecisionsPayload(raw).isValid).toBe(false);
  });

  it.each(["{", "", "null", "[]", '{"input":42}', '{"questions":{}}', '{"questions":[{"type":"noul"}]}'])(
    "rejects unsupported form shapes without replacing the JSON: %s",
    (raw) => {
      expect(readDecisionsForm(raw)).toBeUndefined();
    },
  );
});

describe("retypeDecisionQuestion", () => {
  const choice: DecisionsFormQuestion = {
    type: "choice",
    name: "route",
    instructions: "Where should it go?",
    choices: [{ value: false }, { value: "other" }],
  };

  it("keeps choices unchanged when the type does not change", () => {
    expect(retypeDecisionQuestion(choice, "choice")).toEqual(choice);
  });

  it("removes type-specific fields while preserving names, instructions and extras", () => {
    const score = retypeDecisionQuestion({ ...choice, custom: "keep" }, "score");
    const expectedScore = {
      type: "score",
      name: "route",
      instructions: "Where should it go?",
      custom: "keep",
      levels: [{ label: "0" }, { label: "1" }],
    };
    expect(score).toEqual(expectedScore);
    const predicate = retypeDecisionQuestion(score, "predicate");
    const expectedPredicate = {
      type: "predicate",
      name: "route",
      instructions: "Where should it go?",
      custom: "keep",
    };
    expect(predicate).toEqual(expectedPredicate);
    expect(retypeDecisionQuestion(predicate, "choice")).toEqual({
      ...predicate,
      type: "choice",
      choices: [{ value: "option_1" }, { value: "option_2" }],
    });
    expect(choice.choices).toEqual([{ value: false }, { value: "other" }]);
  });
});
