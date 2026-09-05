import { z } from "zod/v4";
import { isBudgetAddressable } from "./AccessGroupBudgetColumns";

export const createAccessGroupSchema = (existingGroups: ReadonlySet<string>) =>
  z.object({
    access_group: z
      .string()
      .trim()
      .min(1, "Enter a name for the access group")
      .refine(isBudgetAddressable, 'A group name cannot contain "/"')
      .refine((name) => !existingGroups.has(name), "An access group with this name already exists"),
    model_names: z.array(z.string()).min(1, "Pick at least one model"),
  });

export type CreateAccessGroupFormValues = z.input<ReturnType<typeof createAccessGroupSchema>>;
