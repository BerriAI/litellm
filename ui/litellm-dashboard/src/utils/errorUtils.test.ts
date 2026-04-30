import { describe, it, expect } from "vitest";
import { extractErrorMessage } from "./errorUtils";

describe("extractErrorMessage", () => {
  it("should return the message from an Error instance", () => {
    expect(extractErrorMessage(new Error("Something broke"))).toBe("Something broke");
  });

  it("should return detail when it is a string", () => {
    expect(extractErrorMessage({ detail: "Not found" })).toBe("Not found");
  });

  it("should join msg fields from a FastAPI 422 detail array", () => {
    const err = {
      detail: [
        { msg: "field required", loc: ["body", "name"], type: "value_error" },
        { msg: "invalid type", loc: ["body", "age"], type: "type_error" },
      ],
    };
    expect(extractErrorMessage(err)).toBe("field required; invalid type");
  });

  it("should extract error from nested detail object", () => {
    expect(extractErrorMessage({ detail: { error: "bad request" } })).toBe("bad request");
  });

  it("should fall back to message property on plain objects", () => {
    expect(extractErrorMessage({ message: "fallback msg" })).toBe("fallback msg");
  });

  it("should JSON.stringify unknown object shapes", () => {
    expect(extractErrorMessage({ foo: "bar" })).toBe('{"foo":"bar"}');
  });

  it("should stringify primitive non-object values", () => {
    expect(extractErrorMessage(42)).toBe("42");
    expect(extractErrorMessage(null)).toBe("null");
    expect(extractErrorMessage(undefined)).toBe("undefined");
  });
});
