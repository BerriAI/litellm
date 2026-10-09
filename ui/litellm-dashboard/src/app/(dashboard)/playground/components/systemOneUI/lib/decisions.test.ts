import { describe, expect, it } from "vitest";
import { validateSystemOnePayload } from "./validatePayload";
import { systemOneResponseSchema } from "./schemas";

const request = {
  model: "configured-decision-model",
  state: { message: "Please help" },
  questions: {
    category: { type: "choice", criteria: { support: { description: "Help" }, other: null } },
    urgent: { type: "noul", criteria: { true: ["An outage"], false: null } },
    severity: { type: "score", instructions: { task: "Rate severity" }, criteria: [["Low"]] },
  },
  provider_option: { enabled: true },
};
const validate = (value: unknown) => validateSystemOnePayload(JSON.stringify(value), "/v1/systemone");

describe("native decisions validation", () => {
  it("accepts structured Jev criteria, optional instructions, and provider extensions without dropping fields", () => {
    expect(validate(request)).toMatchObject({ isValid: true, payload: request, issues: [] });
  });

  it("accepts an omitted model without inserting one", () => {
    const payload = { state: request.state, questions: request.questions };
    expect(validate(payload)).toMatchObject({ isValid: true, payload, issues: [] });
    expect(validate(payload).payload).not.toHaveProperty("model");
  });

  it.each([null, "", "   ", 1])("rejects an invalid explicit model (%s)", (model) => {
    expect(validate({ ...request, model }).isValid).toBe(false);
  });

  it.each([null, 1, true])("rejects a scalar state (%s)", (state) => {
    expect(validate({ ...request, state }).isValid).toBe(false);
  });

  it.each([
    {},
    { "": request.questions.category },
    Object.fromEntries(Array.from({ length: 129 }, (_, i) => [`q${i}`, request.questions.category])),
    { q: { type: "noul" } },
    { q: { type: "noul", criteria: { unexpected: "value" } } },
    { q: { type: "choice", criteria: {} } },
    { q: { type: "choice", criteria: { invalid: 1 } } },
    { q: { type: "choice", criteria: Object.fromEntries(Array.from({ length: 256 }, (_, i) => [`c${i}`, null])) } },
    { q: { type: "score", criteria: [] } },
    { q: { type: "score", criteria: Array(11).fill("level") } },
  ])("rejects invalid questions %#", (questions) => {
    expect(validate({ ...request, questions }).isValid).toBe(false);
  });

  it("allows backend boundary values for score levels and question counts", () => {
    expect(
      validate({ ...request, questions: { q: { type: "score", criteria: Array(10).fill("level") } } }).isValid,
    ).toBe(true);
    expect(
      validate({
        ...request,
        questions: Object.fromEntries(Array.from({ length: 128 }, (_, i) => [`q${i}`, request.questions.category])),
      }).isValid,
    ).toBe(true);
  });

  it("does not loosen legacy TypeSafe request validation", () => {
    expect(validateSystemOnePayload(JSON.stringify(request)).isValid).toBe(false);
  });

  it("parses structured score legends and nullable usage from the native endpoint", () => {
    const response = {
      answers: {
        severity: {
          type: "score",
          score: 0,
          confidence: 1,
          probabilities: { "0": 1 },
          legend: { "0": { description: "Low" } },
        },
        urgent: { type: "noul", noul: 0.2 },
      },
      usage: null,
    };
    expect(systemOneResponseSchema.parse(response)).toEqual(response);
  });
});
