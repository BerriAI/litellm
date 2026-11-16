import { describe, expect, it } from "vitest";
import { uuid } from "./uuid";

const UUID_V4_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe("uuid", () => {
  it("returns a valid RFC 4122 version 4 UUID", () => {
    expect(uuid()).toMatch(UUID_V4_REGEX);
  });

  it("returns distinct values on repeated calls", () => {
    expect(new Set([uuid(), uuid(), uuid()]).size).toBe(3);
  });

  it("falls back to getRandomValues when crypto.randomUUID is unavailable", () => {
    const descriptor = Object.getOwnPropertyDescriptor(crypto, "randomUUID");
    Object.defineProperty(crypto, "randomUUID", {
      value: undefined,
      configurable: true,
    });
    try {
      expect(uuid()).toMatch(UUID_V4_REGEX);
    } finally {
      if (descriptor) {
        Object.defineProperty(crypto, "randomUUID", descriptor);
      }
    }
  });
});
