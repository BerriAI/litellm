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
  isEdit?: boolean,
  callbackType?: "success" | "failure" | "success_and_failure",
  currentCallbackSettings?: Record<string, unknown>
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
    const callbackSettingsEntry = {
      callback_type: "websearch_interception" as const,
      event_types: ["llm_api_success", "llm_api_failure"] as const,
      enabled_providers,
      ...(search_tool_name ? { search_tool_name } : {}),
    };
    const baseSettings: Record<string, unknown> = { ...(currentCallbackSettings || {}) };
    delete baseSettings.websearch_interception;
    return {
      litellm_settings: { callbacks: list },
      callback_settings: { ...baseSettings, websearch_interception: callbackSettingsEntry },
    };
  }
  const type = callbackType ?? "success";
  const eventTypes =
    type === "success"
      ? ["llm_api_success"]
      : type === "failure"
        ? ["llm_api_failure"]
        : ["llm_api_success", "llm_api_failure"];
  const endpoint = (formValues.GENERIC_LOGGER_ENDPOINT as string)?.trim();
  const headersStr = (formValues.GENERIC_LOGGER_HEADERS as string)?.trim();
  const isGenericApi = Boolean(endpoint);
  const callbackSettingsEntry = isGenericApi
    ? {
        callback_type: "generic_api" as const,
        endpoint,
        ...(headersStr ? { headers: parseHeadersString(headersStr) } : {}),
        event_types: eventTypes,
      }
    : { callback_type: callbackName, event_types: eventTypes };
  const mergedCallbackSettings = { ...(currentCallbackSettings || {}), [callbackName]: callbackSettingsEntry };
  return {
    environment_variables: formValues,
    litellm_settings: {
      ...(type === "success" && { success_callback: [callbackName] }),
      ...(type === "failure" && { failure_callback: [callbackName] }),
      ...(type === "success_and_failure" && { callbacks: [callbackName] }),
    },
    callback_settings: mergedCallbackSettings,
  };
}

function parseHeadersString(headersStr: string): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const part of headersStr.split(",")) {
    const colonIdx = part.indexOf(":");
    if (colonIdx > 0) {
      const key = part.slice(0, colonIdx).trim();
      const value = part.slice(colonIdx + 1).trim();
      if (key) headers[key] = value;
    }
  }
  return headers;
}
