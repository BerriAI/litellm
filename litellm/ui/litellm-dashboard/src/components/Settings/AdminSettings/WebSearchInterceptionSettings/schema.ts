import { z } from "zod";

const webSearchInterceptionShape = {
  enabled: z.boolean(),
  enabled_providers: z.array(z.string()),
  search_tool_name: z.string().nullable(),
  max_agentic_loops: z.number().nullable(),
};

export const webSearchInterceptionSchema = z.object(webSearchInterceptionShape);

export type WebSearchInterceptionFormValues = z.output<typeof webSearchInterceptionSchema>;
