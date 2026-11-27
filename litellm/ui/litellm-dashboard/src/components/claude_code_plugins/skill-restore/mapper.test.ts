import { describe, expect, it } from "vitest";
import { toSkillRestorePayload } from "./mapper";
import { skillRestoreSchema } from "./schema";

describe("skill restore payload", () => {
  it("sends the version the admin entered with the current skill's concurrency fields", () => {
    const values = skillRestoreSchema.parse({ newVersion: " 2.1.0 " });
    expect(toSkillRestorePayload(values, { id: "current-skill", updated_at: "2026-01-15T00:00:00Z" })).toEqual({
      expected_skill_id: "current-skill",
      expected_updated_at: "2026-01-15T00:00:00Z",
      version: "2.1.0",
    });
  });

  it("requests an automatic version when the field is blank", () => {
    const values = skillRestoreSchema.parse({ newVersion: "  " });
    expect(toSkillRestorePayload(values, { id: "current-skill" })).toEqual({
      expected_skill_id: "current-skill",
      expected_updated_at: null,
      version: null,
    });
  });
});
