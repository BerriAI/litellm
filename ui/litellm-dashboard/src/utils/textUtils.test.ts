import { describe, it, expect } from "vitest";
import { formatLabel, formItemValidateJSON, truncateString } from "./textUtils";

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

describe("formItemValidateJSON", () => {
  it("should resolve for a valid JSON", async () => {
    const validObj = { a: 1, b: "x", c: true, d: [1, 2], e: { f: "y" } };
    await expect(formItemValidateJSON({}, JSON.stringify(validObj))).resolves.toBeUndefined();
  });

  it("should reject with an error message for invalid JSON", async () => {
    await expect(formItemValidateJSON({}, "invalid JSON")).rejects.toBe("Please enter valid JSON");
  });
});
