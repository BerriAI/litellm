import { z } from "zod";

const semanticFilterShape = {
  enabled: z.boolean(),
  embedding_model: z.string().nullable(),
  top_k: z.number().nullable(),
  similarity_threshold: z.number(),
};

export const mcpSemanticFilterSchema = z.object(semanticFilterShape);

export type MCPSemanticFilterFormValues = z.output<typeof mcpSemanticFilterSchema>;
