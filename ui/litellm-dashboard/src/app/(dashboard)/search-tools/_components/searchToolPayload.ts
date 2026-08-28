export interface SearchToolFormValues {
  search_tool_name: string;
  search_provider: string;
  api_key?: string | null;
  api_base?: string;
  timeout?: string;
  max_retries?: string;
  description?: string | null;
}

export interface SearchToolPayload {
  search_tool_name: string;
  litellm_params: {
    search_provider: string;
    api_key: string | null | undefined;
    api_base: string | undefined;
    timeout: number | undefined;
    max_retries: number | undefined;
  };
  search_tool_info: { description: string } | undefined;
}

export const buildSearchToolPayload = (values: SearchToolFormValues): SearchToolPayload => ({
  search_tool_name: values.search_tool_name,
  litellm_params: {
    search_provider: values.search_provider,
    api_key: values.api_key,
    api_base: values.api_base,
    timeout: values.timeout ? parseFloat(values.timeout) : undefined,
    max_retries: values.max_retries ? parseInt(values.max_retries, 10) : undefined,
  },
  search_tool_info: values.description ? { description: values.description } : undefined,
});
