/**
 * Helpers for building litellm_settings callback payloads (used by Settings when adding/editing callbacks).
 * Extracted for testability.
 */

export function parseEnabledProviders(v: unknown): string[] {
  if (Array.isArray(v)) return v.map((s) => String(s).trim()).filter(Boolean);
  if (typeof v === "string") return v.split(",").map((s) => s.trim()).filter(Boolean);
  return [];
}

export function reconstructCallbacksList(
  currentCallbacks: Array<{ name: string; type?: string; params?: Record<string, unknown> }>
): Array<string | Record<string, unknown>> {
  return currentCallbacks
    .filter((c) => c.type === "success_and_failure")
    .map((c) => (c.params ? { [c.name]: c.params } : c.name));
}

export function buildCallbackPayload(
  formValues: Record<string, unknown>,
  callbackName: string,
  currentCallbacks?: Array<{ name: string; type?: string; params?: Record<string, unknown> }>,
  isEdit?: boolean
): Record<string, unknown> {
  if (callbackName === "websearch_interception") {
    const enabled_providers = parseEnabledProviders(formValues.enabled_providers);
    const search_tool_name = (formValues.search_tool_name as string)?.trim() || undefined;
    const newEntry = {
      websearch_interception: {
        enabled_providers,
        ...(search_tool_name ? { search_tool_name } : {}),
      },
    };
    const baseList = currentCallbacks ? reconstructCallbacksList(currentCallbacks) : [];
    const list = isEdit
      ? baseList.map((item) =>
          typeof item === "object" && item !== null && "websearch_interception" in item
            ? newEntry
            : item
        )
      : [...baseList, newEntry];
    return { litellm_settings: { callbacks: list } };
  }
  return {
    environment_variables: formValues,
    litellm_settings: {
      success_callback: [callbackName],
    },
  };
}
