import type { z } from "zod";
import type { components } from "@/lib/http/schema";
import type { skillRestoreSchema } from "./schema";

export function toSkillRestorePayload(
  { newVersion }: z.output<typeof skillRestoreSchema>,
  skill: Pick<components["schemas"]["SkillHubSkill"], "id" | "updated_at">,
): components["schemas"]["SkillHubRestore"] {
  return {
    expected_skill_id: skill.id,
    expected_updated_at: skill.updated_at ?? null,
    version: newVersion.trim() || null,
  };
}
