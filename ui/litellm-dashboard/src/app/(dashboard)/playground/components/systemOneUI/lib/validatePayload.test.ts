import { describe, expect, it } from "vitest";
import { SYSTEM_ONE_EXAMPLE } from "./example";
import { validateSystemOnePayload } from "./validatePayload";

const requestWith = (fields: Record<string, unknown> = {}) =>
  JSON.stringify({
    state: "A support request",
    questions: { category: { type: "choice", instructions: "Route the message", criteria: { support: "Help" } } },
    ...fields,
  });

describe("validateSystemOnePayload", () => {
  it("rejects empty input", () => {
    expect(validateSystemOnePayload(" ").issues).toContainEqual({
      path: "root",
      message: "Payload cannot be empty.",
      severity: "error",
    });
  });

  it("reports JSON syntax errors", () => {
    expect(validateSystemOnePayload("{").issues[0]?.path).toBe("syntax");
  });

  it("requires an object root", () => {
    expect(validateSystemOnePayload("[]").issues[0]?.path).toBe("root");
  });

  it("requires the state key while accepting null state", () => {
    expect(validateSystemOnePayload(requestWith({ state: undefined })).issues).toContainEqual(
      expect.objectContaining({ path: "state", severity: "error" }),
    );
    expect(validateSystemOnePayload(requestWith({ state: null })).isValid).toBe(true);
  });

  it.each([null, "", 123])("requires a non-empty string model when present: %s", (model) => {
    expect(validateSystemOnePayload(requestWith({ model })).issues).toContainEqual(
      expect.objectContaining({ path: "model", severity: "error" }),
    );
  });

  it("requires questions", () => {
    expect(validateSystemOnePayload(JSON.stringify({ state: null })).issues).toContainEqual(
      expect.objectContaining({ path: "questions", severity: "error" }),
    );
  });

  it.each([null, [], "questions", {}])("requires a non-empty questions object: %s", (questions) => {
    expect(validateSystemOnePayload(requestWith({ questions })).issues).toContainEqual(
      expect.objectContaining({ path: "questions", severity: "error" }),
    );
  });

  it.each([null, [], "question"])("requires each question to be an object: %s", (question) => {
    expect(validateSystemOnePayload(requestWith({ questions: { invalid: question } })).issues).toContainEqual(
      expect.objectContaining({ path: "questions.invalid", severity: "error" }),
    );
  });

  it("requires a supported question type", () => {
    expect(
      validateSystemOnePayload(requestWith({ questions: { invalid: { type: "other", instructions: "Do this" } } }))
        .issues,
    ).toContainEqual(expect.objectContaining({ path: "questions.invalid.type", severity: "error" }));
  });

  it("requires instructions", () => {
    expect(validateSystemOnePayload(requestWith({ questions: { category: { type: "noul" } } })).issues).toContainEqual(
      expect.objectContaining({ path: "questions.category.instructions", severity: "error" }),
    );
  });

  it("requires instructions to be a string", () => {
    const result = validateSystemOnePayload(
      requestWith({
        questions: { category: { type: "choice", instructions: { text: "Route" }, criteria: { support: "Help" } } },
      }),
    );
    expect(result.isValid).toBe(false);
    expect(result.issues).toContainEqual(
      expect.objectContaining({ path: "questions.category.instructions", severity: "error" }),
    );
  });

  it.each([undefined, null, [], "criteria"])("requires choice criteria as an object: %s", (criteria) => {
    expect(
      validateSystemOnePayload(
        requestWith({ questions: { category: { type: "choice", instructions: "Route", criteria } } }),
      ).issues,
    ).toContainEqual(expect.objectContaining({ path: "questions.category.criteria", severity: "error" }));
  });

  it("requires choice descriptions to be strings", () => {
    const result = validateSystemOnePayload(
      requestWith({
        questions: { category: { type: "choice", instructions: "Route", criteria: { support: { text: "Help" } } } },
      }),
    );
    expect(result.isValid).toBe(false);
    expect(result.issues).toContainEqual(
      expect.objectContaining({ path: "questions.category.criteria.support", severity: "error" }),
    );
  });

  it.each([{}, Object.fromEntries(Array.from({ length: 256 }, (_, index) => [`option-${index}`, "Description"]))])(
    "requires between 1 and 255 choice criteria",
    (criteria) => {
      expect(
        validateSystemOnePayload(
          requestWith({ questions: { category: { type: "choice", instructions: "Route", criteria } } }),
        ).issues,
      ).toContainEqual(expect.objectContaining({ path: "questions.category.criteria", severity: "error" }));
    },
  );

  it("accepts 1 and 255 choice criteria", () => {
    const one = { first: "First option" };
    const twoHundredFiftyFiveOptions = Object.fromEntries(
      Array.from({ length: 255 }, (_, index) => [`option-${index}`, "Option"]),
    );
    expect(
      validateSystemOnePayload(
        requestWith({ questions: { category: { type: "choice", instructions: "Route", criteria: one } } }),
      ).isValid,
    ).toBe(true);
    expect(
      validateSystemOnePayload(
        requestWith({
          questions: { category: { type: "choice", instructions: "Route", criteria: twoHundredFiftyFiveOptions } },
        }),
      ).isValid,
    ).toBe(true);
  });

  it.each([null, {}, "criteria"])("requires score criteria as an array: %s", (criteria) => {
    expect(
      validateSystemOnePayload(
        requestWith({ questions: { urgency: { type: "score", instructions: "Rate urgency", criteria } } }),
      ).issues,
    ).toContainEqual(expect.objectContaining({ path: "questions.urgency.criteria", severity: "error" }));
  });

  it("requires at least two score levels", () => {
    expect(
      validateSystemOnePayload(
        requestWith({ questions: { urgency: { type: "score", instructions: "Rate urgency", criteria: ["Low"] } } }),
      ).issues,
    ).toContainEqual(expect.objectContaining({ path: "questions.urgency.criteria", severity: "error" }));
  });

  it("requires score levels to be strings", () => {
    const result = validateSystemOnePayload(
      requestWith({
        questions: {
          urgency: { type: "score", instructions: "Rate urgency", criteria: ["Low", { level: "High" }] },
        },
      }),
    );
    expect(result.isValid).toBe(false);
    expect(result.issues).toContainEqual(
      expect.objectContaining({ path: "questions.urgency.criteria.1", severity: "error" }),
    );
  });

  it("warns about more than ten score levels without invalidating the request", () => {
    const result = validateSystemOnePayload(
      requestWith({
        questions: {
          urgency: {
            type: "score",
            instructions: "Rate urgency",
            criteria: Array.from({ length: 11 }, () => "Level"),
          },
        },
      }),
    );
    expect(result.isValid).toBe(true);
    expect(result.issues).toContainEqual(
      expect.objectContaining({ path: "questions.urgency.criteria", severity: "warning" }),
    );
  });

  it("accepts a score question with two levels", () => {
    expect(
      validateSystemOnePayload(
        requestWith({
          questions: { urgency: { type: "score", instructions: "Rate urgency", criteria: ["Low", "High"] } },
        }),
      ).isValid,
    ).toBe(true);
  });

  it.each([null, [], "criteria"])("requires optional noul criteria to be an object: %s", (criteria) => {
    expect(
      validateSystemOnePayload(
        requestWith({ questions: { escalation: { type: "noul", instructions: "Escalate?", criteria } } }),
      ).issues,
    ).toContainEqual(expect.objectContaining({ path: "questions.escalation.criteria", severity: "error" }));
  });

  it("allows omitted noul criteria", () => {
    expect(
      validateSystemOnePayload(requestWith({ questions: { escalation: { type: "noul", instructions: "Escalate?" } } }))
        .isValid,
    ).toBe(true);
  });

  it("requires noul criteria definitions to be strings", () => {
    const result = validateSystemOnePayload(
      requestWith({
        questions: {
          escalation: {
            type: "noul",
            instructions: "Escalate?",
            criteria: { true: { meaning: "yes" }, false: 1 },
          },
        },
      }),
    );
    expect(result.isValid).toBe(false);
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ path: "questions.escalation.criteria.true", severity: "error" }),
        expect.objectContaining({ path: "questions.escalation.criteria.false", severity: "error" }),
      ]),
    );
  });

  it("keeps fields outside the known schema so they reach the upstream model", () => {
    const result = validateSystemOnePayload(
      requestWith({
        temperature: 0,
        questions: {
          category: { type: "choice", instructions: "Route", criteria: { support: "Help" }, weight: 2 },
          escalate: { type: "noul", instructions: "Escalate?", criteria: { true: "Yes", unsure: "Maybe" } },
        },
      }),
    );
    expect(result.payload).toMatchObject({ temperature: 0, questions: { category: { weight: 2 } } });
    expect(result.payload?.questions.escalate).toEqual({
      type: "noul",
      instructions: "Escalate?",
      criteria: { true: "Yes", unsure: "Maybe" },
    });
  });

  it("accepts the built-in example without issues", () => {
    expect(validateSystemOnePayload(JSON.stringify(SYSTEM_ONE_EXAMPLE))).toMatchObject({ isValid: true, issues: [] });
  });
});
