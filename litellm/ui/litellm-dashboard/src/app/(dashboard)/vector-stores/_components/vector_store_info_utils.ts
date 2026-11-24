import { z } from "zod";

import { VectorStore } from "@/components/vector_store_management/types";
import i18n from "@/i18n/i18n";

const vectorStoreEditShape = {
  vector_store_id: z.string().min(1, { error: () => i18n.t("vectorStores.info.errors.idRequired") }),
  vector_store_name: z.string().nullish(),
  vector_store_description: z.string().nullish(),
  custom_llm_provider: z.string().min(1, { error: () => i18n.t("vectorStores.info.errors.providerRequired") }),
  litellm_credential_name: z.string().nullable().optional(),
  api_base: z.string().optional(),
  api_key: z.string().optional(),
  embedding_model: z.string().optional(),
};

export const vectorStoreEditSchema = z.object(vectorStoreEditShape).superRefine((values, context) => {
  if (values.custom_llm_provider.toLowerCase() !== "milvus") return;

  if (!values.api_base?.trim()) {
    context.addIssue({
      code: "custom",
      path: ["api_base"],
      message: i18n.t("vectorStores.form.errors.inputField", {
        field: i18n.t("vectorStores.providerFields.labels.apiBase"),
      }),
    });
  }
  if (!values.embedding_model?.trim()) {
    context.addIssue({
      code: "custom",
      path: ["embedding_model"],
      message: i18n.t("vectorStores.form.errors.inputField", {
        field: i18n.t("vectorStores.providerFields.labels.embeddingModel"),
      }),
    });
  }
});

export type VectorStoreEditValues = z.output<typeof vectorStoreEditSchema>;

export const EMPTY_VECTOR_STORE_EDIT_VALUES: VectorStoreEditValues = {
  vector_store_id: "",
  custom_llm_provider: "",
};

export const parseLitellmParams = (value: VectorStore["litellm_params"]): Record<string, unknown> => {
  if (typeof value === "string") {
    try {
      const parsed: unknown = JSON.parse(value);
      return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : {};
    } catch {
      return {};
    }
  }
  return value ?? {};
};

export const toVectorStoreEditValues = (vectorStore: VectorStore): VectorStoreEditValues => {
  const params = parseLitellmParams(vectorStore.litellm_params);
  return {
    vector_store_id: vectorStore.vector_store_id,
    vector_store_name: vectorStore.vector_store_name,
    vector_store_description: vectorStore.vector_store_description,
    custom_llm_provider: vectorStore.custom_llm_provider?.toLowerCase() ?? "",
    litellm_credential_name: vectorStore.litellm_credential_name,
    api_base: typeof params.api_base === "string" ? params.api_base : undefined,
    api_key: "",
    embedding_model: typeof params.litellm_embedding_model === "string" ? params.litellm_embedding_model : undefined,
  };
};

export const buildVectorStoreLitellmParams = (values: VectorStoreEditValues): Record<string, unknown> | undefined => {
  if (values.custom_llm_provider.toLowerCase() !== "milvus") return undefined;

  return {
    api_base: values.api_base,
    litellm_embedding_model: values.embedding_model,
    ...(values.api_key ? { api_key: values.api_key } : {}),
  };
};
