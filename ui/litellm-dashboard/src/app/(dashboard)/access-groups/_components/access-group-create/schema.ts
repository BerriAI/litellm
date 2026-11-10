import { z } from "zod";

import i18n from "@/i18n/i18n";

export const accessGroupCreateSchema = z.object({
  name: z.string().refine((value) => value.trim() !== "", { error: () => i18n.t("accessGroups.form.name.required") }),
  description: z.string(),
  modelIds: z.array(z.string()),
  mcpServerIds: z.array(z.string()),
  mcpToolPermissions: z.record(z.string(), z.array(z.string())),
  agentIds: z.array(z.string()),
  skillNames: z.array(z.string()),
});

export type AccessGroupCreateFormValues = z.output<typeof accessGroupCreateSchema>;
