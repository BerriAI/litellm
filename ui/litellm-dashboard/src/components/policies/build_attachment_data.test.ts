import { describe, expect, it } from "vitest";
import { buildAttachmentData } from "./build_attachment_data";

describe("buildAttachmentData", () => {
  describe("when scope is global", () => {
    it("should set scope to '*'", () => {
      const result = buildAttachmentData({ policy_name: "my-policy" }, "global");
      expect(result.scope).toBe("*");
    });

    it("should not include teams, keys, models, or tags even when provided", () => {
      const result = buildAttachmentData(
        { policy_name: "my-policy", teams: ["team-a"], keys: ["key-1"], models: ["gpt-4"], tags: ["prod"] },
        "global"
      );
      expect(result.teams).toBeUndefined();
      expect(result.keys).toBeUndefined();
      expect(result.models).toBeUndefined();
      expect(result.tags).toBeUndefined();
    });

    it("should include the policy_name", () => {
      const result = buildAttachmentData({ policy_name: "rate-limit-policy" }, "global");
      expect(result.policy_name).toBe("rate-limit-policy");
    });
  });

  describe("when scope is specific", () => {
    it("should not set scope field", () => {
      const result = buildAttachmentData({ policy_name: "my-policy" }, "specific");
      expect(result.scope).toBeUndefined();
    });

    it("should include teams when provided", () => {
      const result = buildAttachmentData({ policy_name: "p", teams: ["team-a", "team-b"] }, "specific");
      expect(result.teams).toEqual(["team-a", "team-b"]);
    });

    it("should include keys when provided", () => {
      const result = buildAttachmentData({ policy_name: "p", keys: ["sk-abc"] }, "specific");
      expect(result.keys).toEqual(["sk-abc"]);
    });

    it("should include models when provided", () => {
      const result = buildAttachmentData({ policy_name: "p", models: ["gpt-4", "claude-3"] }, "specific");
      expect(result.models).toEqual(["gpt-4", "claude-3"]);
    });

    it("should include tags when provided", () => {
      const result = buildAttachmentData({ policy_name: "p", tags: ["prod", "us-east"] }, "specific");
      expect(result.tags).toEqual(["prod", "us-east"]);
    });

    it("should omit teams when the array is empty", () => {
      const result = buildAttachmentData({ policy_name: "p", teams: [] }, "specific");
      expect(result.teams).toBeUndefined();
    });

    it("should omit keys when the array is empty", () => {
      const result = buildAttachmentData({ policy_name: "p", keys: [] }, "specific");
      expect(result.keys).toBeUndefined();
    });

    it("should omit models when the array is empty", () => {
      const result = buildAttachmentData({ policy_name: "p", models: [] }, "specific");
      expect(result.models).toBeUndefined();
    });

    it("should omit tags when the array is empty", () => {
      const result = buildAttachmentData({ policy_name: "p", tags: [] }, "specific");
      expect(result.tags).toBeUndefined();
    });

    it("should omit all scope fields when none are provided", () => {
      const result = buildAttachmentData({ policy_name: "p" }, "specific");
      expect(result.teams).toBeUndefined();
      expect(result.keys).toBeUndefined();
      expect(result.models).toBeUndefined();
      expect(result.tags).toBeUndefined();
    });
  });
});
