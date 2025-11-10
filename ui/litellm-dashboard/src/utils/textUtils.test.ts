import { describe, it, expect } from "vitest";
import { formatLabel } from "./textUtils";

describe("textUtils", () => {
  it("should format label", () => {
    expect(formatLabel("test_label")).toBe("Test Label");
  });
});
