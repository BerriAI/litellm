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
    expect(validateSystemOnePayload("[]").issues).toContainEqual(
      expect.objectContaining({ path: "root", message: "Payload must be a JSON object.", severity: "error" }),
    );
  });

  it("requires the state key while accepting null state", () => {
    expect(validateSystemOnePayload(requestWith({ state: undefined })).issues).toContainEqual(
      expect.objectContaining({
        path: "state",
        message: "Required property 'state' is missing.",
        severity: "error",
      }),
    );
    expect(validateSystemOnePayload(requestWith({ state: null })).isValid).toBe(true);
  });

  it.each([null, "", 123])("requires a non-empty string model when present: %s", (model) => {
    expect(validateSystemOnePayload(requestWith({ model })).issues).toContainEqual(
      expect.objectContaining({ path: "model", message: "Model must be a non-empty string.", severity: "error" }),
    );
  });

  it("requires questions", () => {
    expect(validateSystemOnePayload(JSON.stringify({ state: null })).issues).toContainEqual(
      expect.objectContaining({
        path: "questions",
        message: "Required property 'questions' is missing.",
        severity: "error",
      }),
    );
  });

  it.each([
    [null, "Questions must be a non-empty object."],
    [[], "Questions must be a non-empty object."],
    ["questions", "Questions must be a non-empty object."],
    [{}, "At least one question is required."],
  ])("requires a non-empty questions object: %s", (questions, message) => {
    expect(validateSystemOnePayload(requestWith({ questions })).issues).toContainEqual(
      expect.objectContaining({ path: "questions", message, severity: "error" }),
    );
  });

  it.each([null, [], "question"])("requires each question to be an object: %s", (question) => {
    expect(validateSystemOnePayload(requestWith({ questions: { invalid: question } })).issues).toContainEqual(
      expect.objectContaining({
        path: "questions.invalid",
        message: "Question must be an object.",
        severity: "error",
      }),
    );
  });

  it("requires a supported question type", () => {
    expect(
      validateSystemOnePayload(requestWith({ questions: { invalid: { type: "other", instructions: "Do this" } } }))
        .issues,
    ).toContainEqual(
      expect.objectContaining({
        path: "questions.invalid.type",
        message: "Question type must be choice, noul, or score.",
        severity: "error",
      }),
    );
  });

  it("requires instructions", () => {
    expect(validateSystemOnePayload(requestWith({ questions: { category: { type: "noul" } } })).issues).toContainEqual(
      expect.objectContaining({
        path: "questions.category.instructions",
        message: "Required property 'instructions' is missing.",
        severity: "error",
      }),
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
      expect.objectContaining({
        path: "questions.category.instructions",
        message: "Instructions must be a string.",
        severity: "error",
      }),
    );
  });

  it.each([undefined, null, [], "criteria"])("requires choice criteria as an object: %s", (criteria) => {
    expect(
      validateSystemOnePayload(
        requestWith({ questions: { category: { type: "choice", instructions: "Route", criteria } } }),
      ).issues,
    ).toContainEqual(
      expect.objectContaining({
        path: "questions.category.criteria",
        message: "Choice criteria must be an object mapping labels to descriptions.",
        severity: "error",
      }),
    );
  });

  it("requires choice descriptions to be strings", () => {
    const result = validateSystemOnePayload(
      requestWith({
        questions: { category: { type: "choice", instructions: "Route", criteria: { support: { text: "Help" } } } },
      }),
    );
    expect(result.isValid).toBe(false);
    expect(result.issues).toContainEqual(
      expect.objectContaining({
        path: "questions.category.criteria.support",
        message: "Choice descriptions must be strings.",
        severity: "error",
      }),
    );
  });

  it.each([{}, Object.fromEntries(Array.from({ length: 256 }, (_, index) => [`option-${index}`, "Description"]))])(
    "requires between 1 and 255 choice criteria",
    (criteria) => {
      expect(
        validateSystemOnePayload(
          requestWith({ questions: { category: { type: "choice", instructions: "Route", criteria } } }),
        ).issues,
      ).toContainEqual(
        expect.objectContaining({
          path: "questions.category.criteria",
          message: "Choice criteria must contain between 1 and 255 options.",
          severity: "error",
        }),
      );
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
    ).toContainEqual(
      expect.objectContaining({
        path: "questions.urgency.criteria",
        message: "Score criteria must be an array of levels.",
        severity: "error",
      }),
    );
  });

  it("requires at least two score levels", () => {
    expect(
      validateSystemOnePayload(
        requestWith({ questions: { urgency: { type: "score", instructions: "Rate urgency", criteria: ["Low"] } } }),
      ).issues,
    ).toContainEqual(
      expect.objectContaining({
        path: "questions.urgency.criteria",
        message: "Score criteria must contain at least 2 levels.",
        severity: "error",
      }),
    );
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
      expect.objectContaining({
        path: "questions.urgency.criteria.1",
        message: "Score levels must be strings.",
        severity: "error",
      }),
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
      expect.objectContaining({
        path: "questions.urgency.criteria",
        message: "More than 10 score levels may reduce result quality.",
        severity: "warning",
      }),
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
    ).toContainEqual(
      expect.objectContaining({
        path: "questions.escalation.criteria",
        message: "Noul criteria, when provided, must be an object.",
        severity: "error",
      }),
    );
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
        expect.objectContaining({
          path: "questions.escalation.criteria.true",
          message: "Noul true criteria must be a string.",
          severity: "error",
        }),
        expect.objectContaining({
          path: "questions.escalation.criteria.false",
          message: "Noul false criteria must be a string.",
          severity: "error",
        }),
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
