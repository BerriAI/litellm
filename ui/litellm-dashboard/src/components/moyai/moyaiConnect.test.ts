import { describe, expect, it } from "vitest";
import { normalizeMoyaiUrl } from "./moyaiConnect";

describe("normalizeMoyaiUrl", () => {
  it("trims whitespace and strips trailing slashes", () => {
    expect(normalizeMoyaiUrl("  https://moyai.example.com/  ")).toBe("https://moyai.example.com");
  });

  it("keeps paths and ports", () => {
    expect(normalizeMoyaiUrl("http://localhost:8080/moyai/")).toBe("http://localhost:8080/moyai");
  });

  it.each([
    "javascript:alert(1)",
    "ftp://moyai.example.com",
    "not a url",
    "https://",
    "https://user:pass@moyai.example.com",
    "https://user@moyai.example.com",
    "   ",
  ])("rejects %s", (input) => {
    expect(normalizeMoyaiUrl(input)).toBeNull();
  });
});
