import { describe, it, expect } from "vitest";
import { formatLabel, truncateString } from "./textUtils";

describe("formatLabel", () => {
  it("should format label", () => {
    expect(formatLabel("test_label")).toBe("Test Label");
  });
});

describe("truncateString", () => {
  it("should truncate a string", () => {
    expect(truncateString("Hello, world!", 5)).toBe("Hello...");
  });

  it("should return the original string if it is less than the max length", () => {
    expect(truncateString("Hello, world!", 20)).toBe("Hello, world!");
  });
});
