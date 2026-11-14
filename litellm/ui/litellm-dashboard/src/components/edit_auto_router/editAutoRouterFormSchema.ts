import { z } from "zod";

import i18n from "@/i18n/i18n";

const sharedShape = {
  auto_router_name: z
    .string()
    .min(1, { error: () => i18n.t("editAutoRouter.editAutoRouterModal.errors.nameRequired") }),
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
    .pipe(
      z
        .string({ error: () => i18n.t("editAutoRouter.editAutoRouterModal.errors.defaultModelRequired") })
        .min(1, { error: () => i18n.t("editAutoRouter.editAutoRouterModal.errors.defaultModelRequired") }),
    ),
  auto_router_embedding_model: z
    .string()
    .nullable()
    .pipe(
      z
        .string({ error: () => i18n.t("editAutoRouter.editAutoRouterModal.errors.embeddingModelRequired") })
        .min(1, { error: () => i18n.t("editAutoRouter.editAutoRouterModal.errors.embeddingModelRequired") }),
    ),
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
