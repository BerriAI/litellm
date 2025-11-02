import { describe, expect, it } from "vitest";
import { mapEmptyStringToNull } from "./keyUpdateUtils";

describe("keyUpdateUtils", () => {
  it("should map empty string to null", () => {
    expect(mapEmptyStringToNull("")).toBeNull();
  });

  it("should return the original string otherwise", () => {
    expect(mapEmptyStringToNull("500")).toBe("500");
  });
});
