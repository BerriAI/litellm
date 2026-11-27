import { z } from "zod";

const toolSearchShape = {
  embedding_model: z.string().nullable(),
  top_k: z.number(),
  similarity_threshold: z.number(),
  core_tools_text: z.string(),
};

export const toolSearchSchema = z.object(toolSearchShape);

export type ToolSearchFormValues = z.output<typeof toolSearchSchema>;
