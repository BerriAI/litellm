import { describe, expect, it } from "vitest";
import { validateSystemOnePayload, validateOpenAIDecisionsPayload } from "./validatePayload";
import { systemOneResponseSchema } from "./schemas";
import { openAIDecisionsExample } from "./example";

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

describe("/v1/decisions validation", () => {
  const example = openAIDecisionsExample("jev-latest");
  const validateOpenAI = (value: unknown) => validateOpenAIDecisionsPayload(JSON.stringify(value));

  it("keeps input, question arrays, provider extensions and optional model omission intact", () => {
    const payload = { input: example.input, questions: example.questions, provider_option: { enabled: true } };
    expect(validateOpenAI(payload)).toEqual({ isValid: true, payload, issues: [] });
    expect(validateOpenAI(example).isValid).toBe(true);
  });

  it("accepts multimodal messages and boolean choices without converting them to strings", () => {
    const payload = {
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "Does this show an error?" },
            { type: "input_image", image_url: "data:image/png;base64,example" },
          ],
        },
      ],
      questions: [{ type: "choice", instructions: "Is this an error?", choices: [{ value: true }, { value: false }] }],
    };
    expect(validateOpenAI(payload)).toEqual({ isValid: true, payload, issues: [] });
  });

  it.each([
    request,
    { ...example, questions: [] },
    { ...example, questions: request.questions },
    { ...example, input: {} },
    { ...example, questions: [{ type: "noul", instructions: "Test" }] },
    { ...example, questions: [{ type: "predicate" }] },
    { ...example, questions: [{ type: "predicate", instructions: "Test", criteria: {} }] },
    { ...example, questions: [{ type: "choice", instructions: "Test", choices: [{ value: true }] }] },
    {
      ...example,
      questions: [{ type: "choice", instructions: "Test", choices: [{ value: true }, { value: "true" }] }],
    },
    { ...example, questions: [{ type: "score", instructions: "Test", levels: [{ label: "Low" }] }] },
  ])("rejects the wrong endpoint's shape and invalid question definitions %#", (payload) => {
    expect(validateOpenAI(payload).isValid).toBe(false);
  });

  it("rejects blank and malformed JSON without throwing", () => {
    expect(validateOpenAIDecisionsPayload(" ").issues[0]?.path).toBe("root");
    expect(validateOpenAIDecisionsPayload("{").issues[0]?.path).toBe("syntax");
  });
});
