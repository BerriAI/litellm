import type { components } from "@/lib/http/schema";
import type { MCPSemanticFilterFormValues } from "./schema";

type Settings = components["schemas"]["MCPSemanticFilterSettings"];
export type StoredValues = Partial<Settings>;
type NullableFields = "embedding_model" | "top_k";
type Payload = Omit<Settings, NullableFields> & Pick<MCPSemanticFilterFormValues, NullableFields>;

export const DEFAULT_FORM_VALUES: MCPSemanticFilterFormValues = {
  enabled: false,
  embedding_model: "text-embedding-3-small",
  top_k: 10,
  similarity_threshold: 0.3,
};

export const TOP_K_MIN = 1;
export const TOP_K_MAX = 100;

export const clampTopK = (value: number | null): number | null =>
  value === null ? null : Math.min(TOP_K_MAX, Math.max(TOP_K_MIN, value));

export const parseTopK = (raw: string, rawAsNumber: number): number | null =>
  raw === "" || Number.isNaN(rawAsNumber) ? null : rawAsNumber;

export const storedValuesToForm = (values: StoredValues): MCPSemanticFilterFormValues => ({
  enabled: values.enabled ?? DEFAULT_FORM_VALUES.enabled,
  embedding_model: values.embedding_model ?? DEFAULT_FORM_VALUES.embedding_model,
  top_k: values.top_k ?? DEFAULT_FORM_VALUES.top_k,
  similarity_threshold: values.similarity_threshold ?? DEFAULT_FORM_VALUES.similarity_threshold,
});

export const formToPayload = (values: MCPSemanticFilterFormValues): Payload => values;
