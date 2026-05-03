export interface FetchToolLiteLLMParams {
  fetch_provider: string;
  api_key?: string;
  api_base?: string;
  timeout?: number;
  max_retries?: number;
  [key: string]: any;
}

export interface FetchToolInfo {
  description?: string;
  [key: string]: any;
}

export interface FetchTool {
  fetch_tool_id?: string;
  fetch_tool_name: string;
  litellm_params: FetchToolLiteLLMParams;
  fetch_tool_info?: FetchToolInfo;
  created_at?: string;
  updated_at?: string;
  is_from_config?: boolean;
}

export interface FetchToolsResponse {
  fetch_tools: FetchTool[];
}

export interface AvailableFetchProvider {
  provider_name: string;
  ui_friendly_name: string;
}

export interface AvailableFetchProvidersResponse {
  providers: AvailableFetchProvider[];
}
