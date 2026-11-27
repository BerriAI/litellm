import { z } from "zod";

export const modelGroupAliasSchema = z.object({
  aliasName: z.string().min(1),
  targetModelGroup: z.string().min(1),
});

export type ModelGroupAliasFields = z.infer<typeof modelGroupAliasSchema>;
