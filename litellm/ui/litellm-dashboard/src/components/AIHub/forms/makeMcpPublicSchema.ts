import { z } from "zod";

export const makeMcpPublicSchema = z.object({
  serverIds: z.array(z.string()),
});

export type MakeMcpPublicValues = z.output<typeof makeMcpPublicSchema>;
