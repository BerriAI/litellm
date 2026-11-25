import type { WebSearchInterceptionSettings } from "@/components/networking";
import type { WebSearchInterceptionFormValues } from "./schema";

type StoredValues = Partial<WebSearchInterceptionSettings>;

export const parseLoops = (raw: string, rawAsNumber: number): number | null =>
  raw === "" || Number.isNaN(rawAsNumber) ? null : rawAsNumber;

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((entry) => typeof entry === "string");

export const toStoredValues = (raw: Readonly<Record<string, unknown>>): StoredValues => ({
  enabled: typeof raw.enabled === "boolean" ? raw.enabled : undefined,
  enabled_providers: isStringArray(raw.enabled_providers) ? raw.enabled_providers : undefined,
  search_tool_name: typeof raw.search_tool_name === "string" ? raw.search_tool_name : null,
  max_agentic_loops: typeof raw.max_agentic_loops === "number" ? raw.max_agentic_loops : null,
});

export const storedValuesToForm = (values: StoredValues): WebSearchInterceptionFormValues => ({
  enabled: values.enabled ?? false,
  enabled_providers: values.enabled_providers ?? [],
  search_tool_name: values.search_tool_name ?? null,
  max_agentic_loops: values.max_agentic_loops ?? null,
});

export const formToPayload = (values: WebSearchInterceptionFormValues): WebSearchInterceptionSettings => values;
