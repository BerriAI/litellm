import { describe, expect, it } from "vitest";
import { streamErrorMessage } from "./streamErrorMessage";

describe("streamErrorMessage", () => {
  it("returns the message of an error object", () => {
    expect(streamErrorMessage({ error: { message: "blocked by guardrail", code: "400" } })).toBe(
      "blocked by guardrail",
    );
  });

  it("returns a string error as is", () => {
    expect(streamErrorMessage({ error: "upstream cut" })).toBe("upstream cut");
  });

  it("serializes an error object without a string message", () => {
    expect(streamErrorMessage({ error: { code: 500 } })).toBe('{"code":500}');
  });

  it.each([[{ choices: [{ delta: { content: "hi" } }] }], [{ error: null }], [7], [null]])(
    "returns null for a frame that carries no error: %j",
    (frame) => {
      expect(streamErrorMessage(frame)).toBeNull();
    },
  );
});
