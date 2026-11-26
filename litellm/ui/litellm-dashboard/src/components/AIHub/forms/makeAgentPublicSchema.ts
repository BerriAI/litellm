import { z } from "zod";

export const makeAgentPublicSchema = z.object({
  agentIds: z.array(z.string()).min(1),
});

export type MakeAgentPublicValues = z.output<typeof makeAgentPublicSchema>;
