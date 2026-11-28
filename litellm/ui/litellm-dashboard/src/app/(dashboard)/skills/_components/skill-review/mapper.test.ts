import { describe, expect, it } from "vitest";
import { toSkillReviewPayload } from "./mapper";
import { skillReviewSchema } from "./schema";

describe("skill review payload", () => {
  it("does not send the rejection note when approving", () => {
    expect(toSkillReviewPayload(skillReviewSchema.parse({ reason: "Not used" }), "approve", 3)).toEqual({
      expected_revision: 3,
    });
  });

  it.each([
    ["  Please update the docs.  ", "Please update the docs."],
    ["  ", null],
  ])("sends the trimmed rejection note or null, preserving the legacy revision fallback", (reason, expected) => {
    expect(toSkillReviewPayload(skillReviewSchema.parse({ reason }), "reject", undefined)).toEqual({
      expected_revision: 0,
      reason: expected,
    });
  });
});
