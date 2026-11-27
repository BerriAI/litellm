import type { z } from "zod";
import type { components } from "@/lib/http/schema";
import type { skillReviewSchema } from "./schema";

export function toSkillReviewPayload(
  { reason }: z.output<typeof skillReviewSchema>,
  action: "approve" | "reject",
  revision: components["schemas"]["SkillHubSubmission"]["revision"],
): components["schemas"]["SkillHubRejection"] {
  return {
    expected_revision: revision ?? 0,
    ...(action === "reject" ? { reason: reason.trim() || null } : {}),
  };
}
