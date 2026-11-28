import { z } from "zod";

export const appResourceTypeSchema = z.enum(["team", "project", "virtual_key"]);
export const appDefinitionSchema = z.object({ draft: appResourceTypeSchema.nullable() });
