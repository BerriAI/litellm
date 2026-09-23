import { z } from "zod/v4";

const sharedShape = {
  auto_router_name: z.string().min(1, "Auto router name is required"),
  model_access_group: z.array(z.string()),
};

const complexityRouterShape = {
  ...sharedShape,
  auto_router_default_model: z
    .string()
    .nullable()
    .transform((value) => value ?? ""),
  auto_router_embedding_model: z
    .string()
    .nullable()
    .transform((value) => value ?? ""),
};

const semanticRouterShape = {
  ...sharedShape,
  auto_router_default_model: z
    .string()
    .nullable()
    .pipe(z.string({ error: "Default model is required" }).min(1, "Default model is required")),
  auto_router_embedding_model: z
    .string()
    .nullable()
    .pipe(z.string({ error: "Embedding model is required" }).min(1, "Embedding model is required")),
};

export const complexityRouterSchema = z.object(complexityRouterShape);
export const semanticRouterSchema = z.object(semanticRouterShape);

export type EditAutoRouterFormValues = z.infer<typeof semanticRouterSchema>;

export const EMPTY_FORM_VALUES: z.input<typeof semanticRouterSchema> = {
  auto_router_name: "",
  auto_router_default_model: null,
  auto_router_embedding_model: null,
  model_access_group: [],
};
