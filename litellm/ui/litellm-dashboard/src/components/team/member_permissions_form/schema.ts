import { z } from "zod";

export const memberPermissionsSchema = z.object({
  team_member_permissions: z.array(z.string()),
});

export type MemberPermissionsFields = z.infer<typeof memberPermissionsSchema>;
