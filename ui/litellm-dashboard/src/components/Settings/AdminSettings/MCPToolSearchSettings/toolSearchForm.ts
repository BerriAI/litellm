import type { MCPToolSearchSettings } from "@/app/(dashboard)/hooks/mcpToolSearchSettings/useMCPToolSearchSettings";

export interface ToolSearchFormValues {
  embedding_model: string;
  top_k: number;
  similarity_threshold: number;
  core_tools_text: string;
}

export const TOP_K_MIN = 1;
export const TOP_K_MAX = 100;

export const DEFAULT_FORM_VALUES: ToolSearchFormValues = {
  embedding_model: "",
  top_k: 5,
  similarity_threshold: 0,
  core_tools_text: "",
};

const isString = (value: unknown): value is string => typeof value === "string";
const isNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

export const parseCoreTools = (text: string): string[] =>
  Array.from(
    new Set(
      text
        .split(/[\n,]/)
        .map((name) => name.trim())
        .filter((name) => name.length > 0),
    ),
  );

export const clampTopK = (value: number): number => Math.min(TOP_K_MAX, Math.max(TOP_K_MIN, Math.round(value)));

export const storedValuesToForm = (values: Record<string, unknown>): ToolSearchFormValues => ({
  embedding_model: isString(values.embedding_model) ? values.embedding_model : DEFAULT_FORM_VALUES.embedding_model,
  top_k: isNumber(values.top_k) ? values.top_k : DEFAULT_FORM_VALUES.top_k,
  similarity_threshold: isNumber(values.similarity_threshold)
    ? values.similarity_threshold
    : DEFAULT_FORM_VALUES.similarity_threshold,
  core_tools_text: Array.isArray(values.core_tools) ? values.core_tools.filter(isString).join("\n") : "",
});

export const formToPayload = (form: ToolSearchFormValues): MCPToolSearchSettings => ({
  embedding_model: form.embedding_model.trim() === "" ? null : form.embedding_model.trim(),
  top_k: clampTopK(form.top_k),
  similarity_threshold: form.similarity_threshold,
  core_tools: parseCoreTools(form.core_tools_text),
});
