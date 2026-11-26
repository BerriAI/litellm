import { z } from "zod";

export const userSearchSchema = z.object({
  user_email: z.string().nullish(),
  user_id: z.string().nullish(),
  role: z.string(),
});

export type UserSearchFields = z.infer<typeof userSearchSchema>;
