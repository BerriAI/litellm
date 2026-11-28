import { z } from "zod";

export const makeModelPublicSchema = z.object({
  modelGroups: z.array(z.string()).min(1),
});

export type MakeModelPublicValues = z.output<typeof makeModelPublicSchema>;
