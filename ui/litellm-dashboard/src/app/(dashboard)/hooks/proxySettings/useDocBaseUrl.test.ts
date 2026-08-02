import { describe, expect, it } from "vitest";
import { resolveDocBaseUrl } from "./useDocBaseUrl";

const FALLBACK = "http://proxy.internal:4000";
const DOC_BASE = "https://gateway.public.example.com";

describe("resolveDocBaseUrl", () => {
  it("prefers the doc base url over the fallback when it is set", () => {
    expect(resolveDocBaseUrl(DOC_BASE, FALLBACK)).toBe(DOC_BASE);
  });

  it("falls back when the doc base url is undefined", () => {
    expect(resolveDocBaseUrl(undefined, FALLBACK)).toBe(FALLBACK);
  });

  it("falls back when the doc base url is null", () => {
    expect(resolveDocBaseUrl(null, FALLBACK)).toBe(FALLBACK);
  });

  it("falls back when the doc base url is an empty string", () => {
    expect(resolveDocBaseUrl("", FALLBACK)).toBe(FALLBACK);
  });

  it("falls back when the doc base url is whitespace only", () => {
    expect(resolveDocBaseUrl("   ", FALLBACK)).toBe(FALLBACK);
  });
});
