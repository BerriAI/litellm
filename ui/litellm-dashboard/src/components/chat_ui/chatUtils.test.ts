import { describe, expect, it } from "vitest";
import { truncateString } from "./chatUtils";

describe("chatUtils", () => {
  describe("truncateString", () => {
    it("should truncate a string", () => {
      expect(truncateString("Hello, world!", 5)).toBe("Hello...");
    });

    it("should return the original string if it is less than the max length", () => {
      expect(truncateString("Hello, world!", 20)).toBe("Hello, world!");
    });
  });
});
