import { z } from "zod/v4";

export const accessGroupFormSchema = z.object({
  name: z.string().refine((value) => value.trim() !== "", "Please enter the access group name"),
  description: z.string(),
  modelIds: z.array(z.string()),
  mcpServerIds: z.array(z.string()),
  agentIds: z.array(z.string()),
});

export type AccessGroupFormValues = z.output<typeof accessGroupFormSchema>;

export const emptyAccessGroupFormValues: AccessGroupFormValues = {
  name: "",
  description: "",
  modelIds: [],
  mcpServerIds: [],
  agentIds: [],
};
