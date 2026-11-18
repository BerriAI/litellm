import { describe, expect, it } from "vitest";
import { payloadModel, withPayloadModel } from "./payloadModel";

describe("payloadModel", () => {
  it("reads the model from a JSON object payload", () => {
    expect(payloadModel('{"model": "jev-latest", "state": "hi"}')).toBe("jev-latest");
  });

  it("is undefined when the model is missing, empty, or not a string", () => {
    expect(payloadModel('{"state": "hi"}')).toBeUndefined();
    expect(payloadModel('{"model": ""}')).toBeUndefined();
    expect(payloadModel('{"model": 3}')).toBeUndefined();
  });

  it("is undefined when the payload is not a JSON object", () => {
    expect(payloadModel("{not json")).toBeUndefined();
    expect(payloadModel('["jev-latest"]')).toBeUndefined();
    expect(payloadModel('"jev-latest"')).toBeUndefined();
  });
});

describe("withPayloadModel", () => {
  it("replaces the model and keeps every other field", () => {
    const next = withPayloadModel(
      '{"model": "old", "state": "hi", "questions": {"q": {"type": "noul"}}}',
      "jev-latest",
    );
    expect(next).toBeDefined();
    expect(JSON.parse(next ?? "")).toEqual({ model: "jev-latest", state: "hi", questions: { q: { type: "noul" } } });
  });

  it("adds the model as the first field when the payload has none", () => {
    const next = withPayloadModel('{"state": "hi"}', "jev-latest");
    expect(Object.keys(JSON.parse(next ?? "{}"))).toEqual(["model", "state"]);
  });

  it("leaves a payload that is not a JSON object alone", () => {
    expect(withPayloadModel("{not json", "jev-latest")).toBeUndefined();
    expect(withPayloadModel("[1, 2]", "jev-latest")).toBeUndefined();
  });
});
