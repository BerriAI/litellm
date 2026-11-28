import { z } from "zod";

export const addTeamSchema = z.object({
  team_id: z.string(),
  role: z.string(),
});

export type AddTeamFields = z.infer<typeof addTeamSchema>;
