import { z } from "zod";

export const makeSkillPublicSchema = z.object({
  skillNames: z.array(z.string()).min(1),
});

export type MakeSkillPublicValues = z.output<typeof makeSkillPublicSchema>;
