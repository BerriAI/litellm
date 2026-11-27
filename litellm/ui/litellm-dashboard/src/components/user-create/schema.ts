import { z } from "zod";

const createUserFields = z.object({
  supply_user_id: z.boolean(),
  user_id: z.string().optional(),
  user_email: z.string().optional(),
  user_role: z.string(),
  team_id: z.string().nullish(),
  organization_ids: z.array(z.string()).optional(),
  metadata: z.string().optional(),
  send_invite_email: z.boolean(),
  models: z.array(z.string()).optional(),
});

export type CreateUserFormValues = z.input<typeof createUserFields>;

export const createUserSchema = z.custom<CreateUserFormValues>((values) => createUserFields.safeParse(values).success);
