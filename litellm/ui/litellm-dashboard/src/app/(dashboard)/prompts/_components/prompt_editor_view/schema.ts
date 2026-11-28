import { z } from "zod";
import type { Tool } from "./types";

const configNumber = z.custom<number>((value) => typeof value === "number");

const promptFields = {
  name: z.string().refine((name) => name.trim() !== ""),
  model: z.string().nullable(),
  config: z.object({
    temperature: configNumber.optional(),
    max_tokens: configNumber.optional(),
    top_p: configNumber.optional(),
  }),
  tools: z.array(z.custom<Tool>()),
  developerMessage: z.string(),
  messages: z.array(z.object({ role: z.string(), content: z.string() })),
  environment: z.string(),
};

export const promptEditorSchema = z.object({ prompt: z.object(promptFields) });
