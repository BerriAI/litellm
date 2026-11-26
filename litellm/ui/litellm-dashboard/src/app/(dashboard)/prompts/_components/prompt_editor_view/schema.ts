import { z } from "zod";
import type { Tool } from "./types";

const promptFields = {
  name: z.string().refine((name) => name.trim() !== ""),
  model: z.string().nullable(),
  config: z.object({
    temperature: z.number().optional(),
    max_tokens: z.number().optional(),
    top_p: z.number().optional(),
  }),
  tools: z.array(z.custom<Tool>()),
  developerMessage: z.string(),
  messages: z.array(z.object({ role: z.string(), content: z.string() })),
  environment: z.string(),
};

export const promptEditorSchema = z.object({ prompt: z.object(promptFields) });
