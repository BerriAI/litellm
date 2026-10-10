import { describe, expect, it } from "vitest";
import { SYSTEM_ONE_EXAMPLE, openAIDecisionsExample } from "./example";
import { formFromJson, formIssues, formToJson, formToPayload, newQuestion, withQuestionType } from "./form";
import { validateSystemOnePayload } from "./validatePayload";

const parse = (raw: string) => {
  const result = formFromJson(raw);
  if (!result.ok) {
    throw new Error(result.reason);
  }
  return result.form;
};

describe("formFromJson", () => {
  it("reads the OpenAI example into choice, yes_no, and score questions", () => {
    const form = parse(JSON.stringify(openAIDecisionsExample("decider")));
    expect(form.model).toBe("decider");
    expect(form.context).toBe(SYSTEM_ONE_EXAMPLE.state);
    expect(form.questions.map((question) => question.type)).toEqual(["choice", "yes_no", "score"]);
    expect(form.questions[0]?.choices[0]).toEqual({
      value: "backend",
      description: "Server, API routing, streaming, and fallbacks",
    });
    expect(form.questions[2]?.levels.map((level) => level.label)).toHaveLength(5);
  });

  it("reads the native example and keeps noul criteria as yes / no descriptions", () => {
    const form = parse(JSON.stringify(SYSTEM_ONE_EXAMPLE));
    const noul = form.questions[1];
    expect(noul?.type).toBe("yes_no");
    expect(noul?.name).toBe("has_repro_steps");
    expect(noul?.yes).toBe(SYSTEM_ONE_EXAMPLE.questions.has_repro_steps.criteria.true);
    expect(noul?.no).toBe(SYSTEM_ONE_EXAMPLE.questions.has_repro_steps.criteria.false);
    expect(form.questions[0]?.choices.map((choice) => choice.value)).toEqual(["backend", "sdk", "ui", "docs"]);
    expect(form.questions[2]?.levels.map((level) => level.label)).toEqual(
      SYSTEM_ONE_EXAMPLE.questions.severity.criteria,
    );
  });

  it("refuses JSON the form cannot show and says which field", () => {
    const withExtra = { ...openAIDecisionsExample("decider"), safety_identifier: "abc" };
    expect(formFromJson(JSON.stringify(withExtra))).toEqual({
      ok: false,
      reason: expect.stringContaining("safety_identifier"),
    });
    const messageInput = { ...openAIDecisionsExample("decider"), input: [{ role: "user", content: "hi" }] };
    expect(formFromJson(JSON.stringify(messageInput))).toMatchObject({
      ok: false,
      reason: expect.stringContaining("input"),
    });
    expect(formFromJson("{")).toMatchObject({ ok: false, reason: expect.stringContaining("Invalid JSON") });
    expect(formFromJson("[]")).toMatchObject({ ok: false, reason: expect.stringContaining("JSON object") });
  });
});

describe("formToPayload", () => {
  const form = parse(JSON.stringify(openAIDecisionsExample("decider")));

  it("round trips the OpenAI example through /v1/decisions without changing it", () => {
    expect(formToPayload(form, "/v1/decisions")).toEqual(openAIDecisionsExample("decider"));
  });

  it("round trips the native example through /v1/systemone without changing it", () => {
    const nativeForm = parse(JSON.stringify(SYSTEM_ONE_EXAMPLE));
    expect(formToPayload(nativeForm, "/v1/systemone")).toEqual(SYSTEM_ONE_EXAMPLE);
    expect(formToPayload(nativeForm, "/typesafe/v1/systemone")).toEqual(SYSTEM_ONE_EXAMPLE);
  });

  it("re-shapes the same form for the other endpoint and the result validates there", () => {
    const native = formToPayload(form, "/v1/systemone");
    expect(native).toMatchObject({
      state: SYSTEM_ONE_EXAMPLE.state,
      questions: {
        has_repro_steps: { type: "noul" },
        severity: { type: "score", criteria: expect.arrayContaining(["Outage or data loss"]) },
        area: { type: "choice", criteria: { backend: "Server, API routing, streaming, and fallbacks" } },
      },
    });
    expect(native).not.toHaveProperty("input");
    expect(validateSystemOnePayload(formToJson(form, "/v1/systemone"), "/v1/systemone").isValid).toBe(true);
    expect(validateSystemOnePayload(formToJson(form, "/v1/decisions"), "/v1/decisions").isValid).toBe(true);
  });

  it("omits model, empty names, and empty descriptions instead of sending blanks", () => {
    const minimal = {
      context: "state",
      questions: [
        {
          ...newQuestion("choice"),
          choices: [
            { value: "a", description: "" },
            { value: "b", description: "B" },
          ],
        },
      ],
    };
    expect(formToPayload(minimal, "/v1/decisions")).toEqual({
      input: "state",
      questions: [{ type: "choice", instructions: "", choices: [{ value: "a" }, { value: "b", description: "B" }] }],
    });
    const yesNo = { context: "s", questions: [{ ...newQuestion("yes_no"), name: "q", instructions: "Is it?" }] };
    expect(formToPayload(yesNo, "/v1/systemone")).toEqual({
      state: "s",
      questions: { q: { type: "noul", instructions: "Is it?" } },
    });
  });
});

describe("withQuestionType", () => {
  it("seeds two blank rows when switching to a type with none and keeps existing rows", () => {
    const score = withQuestionType(newQuestion("choice"), "score");
    expect(score.levels).toHaveLength(2);
    expect(score.choices).toHaveLength(2);
    const back = withQuestionType({ ...score, levels: [{ label: "x", description: "" }] }, "score");
    expect(back.levels).toEqual([{ label: "x", description: "" }]);
  });
});

describe("formIssues", () => {
  const base = parse(JSON.stringify(openAIDecisionsExample("decider")));

  it("is empty for the examples on both endpoints", () => {
    expect(formIssues(base, "/v1/decisions")).toEqual([]);
    expect(formIssues(parse(JSON.stringify(SYSTEM_ONE_EXAMPLE)), "/v1/systemone")).toEqual([]);
  });

  it("requires names on native endpoints only", () => {
    const unnamed = { ...base, questions: base.questions.map((question) => ({ ...question, name: "" })) };
    expect(formIssues(unnamed, "/v1/decisions")).toEqual([]);
    expect(formIssues(unnamed, "/v1/systemone")).toHaveLength(3);
  });

  it("flags duplicate names, blank choice values, duplicate choice values, and blank level labels", () => {
    const [choice, yesNo, score] = base.questions;
    if (!choice || !yesNo || !score) {
      throw new Error("example lost a question");
    }
    const broken = {
      ...base,
      questions: [
        { ...choice, choices: [...choice.choices, { value: "", description: "" }, { value: "sdk", description: "" }] },
        { ...yesNo, name: choice.name },
        { ...score, levels: [...score.levels, { label: " ", description: "" }] },
      ],
    };
    expect(formIssues(broken, "/v1/decisions")).toEqual([
      'Two questions are named "area"',
      "Question 1 has a choice without a value",
      'Question 1 lists the choice "sdk" twice',
      "Question 3 has a level without a label",
    ]);
  });
});
