import { z } from "zod";

const accessGroupCreateSchemaFields = {
  name: z.string().refine((value) => value.trim() !== "", "Please enter the access group name"),
  description: z.string(),
  modelIds: z.array(z.string()),
  mcpServerIds: z.array(z.string()),
  agentIds: z.array(z.string()),
  accessRoutes: z.array(z.string()),
};

export const accessGroupCreateSchema = z.object(accessGroupCreateSchemaFields);

export type AccessGroupCreateFormValues = z.output<typeof accessGroupCreateSchema>;
