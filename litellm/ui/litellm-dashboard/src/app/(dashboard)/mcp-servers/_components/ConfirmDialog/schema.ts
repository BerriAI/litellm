import { z } from "zod";

export const rejectionSchema = z.object({
  reviewNotes: z.string(),
});

export type RejectionValues = z.infer<typeof rejectionSchema>;
