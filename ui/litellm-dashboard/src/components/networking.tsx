// Shared date formatter for daily activity endpoints
export const formatDate = (date: Date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

export const getCallbackConfigsCall = async (accessToken: string) => {
  /**
   * Get callback configuration metadata (logos, params, etc.)
   */
  try {
    return await apiClient.get(`/callbacks/configs`, { accessToken });
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

/**
 * Helper file for calls being made to proxy
 */
import MessageManager from "@/components/molecules/message_manager";
import { clearTokenCookies, getCookie, storeLoginToken } from "@/utils/cookieUtils";
import { decodeToken } from "@/utils/jwtUtils";
import { TagNewRequest, TagUpdateRequest, TagListResponse, TagInfoResponse } from "./tag_management/types";
import { Team } from "./key_team_helpers/key_list";
import { EmailEventSettingsResponse, EmailEventSettingsUpdateRequest } from "./email_events/types";
import type { SkillRegisterRequest } from "./claude_code_plugins/types";
import { jsonFields } from "./common_components/check_openapi_schema";
import NotificationsManager from "./molecules/notifications_manager";
import type { MCPUserEnvVarsStatus } from "./mcp_tools/types";
import type {
  CoordinationRedisSettings,
  CoordinationRedisSettingsResponse,
  CoordinationRedisTestResponse,
} from "@/app/(dashboard)/caching/_components/coordination_redis_settings/types";
import { MCP_TOOLS_PREVIEW_FORBIDDEN_MESSAGE } from "./mcp_tools/constants";
import { createApiClient, deriveErrorMessage } from "@/lib/http/client";
import { resolveApiBase } from "@/lib/http/resolveApiBase";
import {
  registerAuthHeaderNameGetter,
  registerAuthTokenGetter,
  registerBaseUrlGetter,
  registerErrorHandler,
} from "@/lib/http/runtime";
import { serverRootPath, setServerRootPath } from "@/lib/serverRootPath";

export { serverRootPath };

export { deriveErrorMessage };
export { ApiError } from "@/lib/http/client";

const isLocal = process.env.NODE_ENV === "development";
// In dev, if NEXT_PUBLIC_USE_REWRITES=true the Next.js dev server proxies API calls
// to the backend — use relative URLs (null) so rewrites can intercept them.
const resolveDefaultBase = (fallback: string | null): string | null =>
  process.env.NEXT_PUBLIC_BASE_URL
    ? process.env.NEXT_PUBLIC_BASE_URL
    : isLocal && process.env.NEXT_PUBLIC_USE_REWRITES !== "true"
      ? "http://localhost:4000"
      : fallback;
const defaultProxyBaseUrl = resolveDefaultBase(null);
const WORKER_URL_KEY = "litellm_worker_url";
// If a worker URL is in localStorage, use it as the initial proxyBaseUrl.
// This survives page navigation and the sessionStorage.clear() in user_dashboard.
const _rawWorkerUrl = typeof window !== "undefined" ? window.localStorage.getItem(WORKER_URL_KEY) : null;
// Validate stored worker URL — reject non-HTTP schemes to prevent exfiltration
const _initialWorkerUrl = (() => {
  if (!_rawWorkerUrl) return null;
  try {
    const parsed = new URL(_rawWorkerUrl);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") return _rawWorkerUrl;
  } catch {
    /* invalid URL */
  }
  // Invalid URL in storage — clear it
  if (typeof window !== "undefined") window.localStorage.removeItem(WORKER_URL_KEY);
  return null;
})();
export let proxyBaseUrl: string | null = _initialWorkerUrl ?? defaultProxyBaseUrl;
if (isLocal != true) {
  console.log = function () {};
}

const getWindowLocation = () => {
  if (typeof window === "undefined") {
    return null;
  }
  return window.location;
};

const updateProxyBaseUrl = (serverRootPath: string, receivedProxyBaseUrl: string | null = null) => {
  /**
   * Special function for updating the proxy base url. Should only be called by getUiConfig.
   */
  // If a worker URL is in localStorage, don't let getUiConfig overwrite it
  if (typeof window !== "undefined" && window.localStorage.getItem(WORKER_URL_KEY)) {
    return;
  }
  proxyBaseUrl = resolveApiBase({
    explicitBase: receivedProxyBaseUrl || resolveDefaultBase(getWindowLocation()?.origin ?? null),
    serverRootPath,
  });
};

const updateServerRootPath = (receivedServerRootPath: string) => {
  setServerRootPath(receivedServerRootPath);
};

export const getProxyBaseUrl = (): string => {
  if (proxyBaseUrl) {
    return proxyBaseUrl;
  }
  const browserLocation = getWindowLocation();
  return browserLocation?.origin ?? "";
};

/**
 * Switch API calls to point at a worker (or back to the control plane).
 * Persists to localStorage so it survives page navigation and the
 * sessionStorage.clear() in user_dashboard. Also updates the module-level
 * proxyBaseUrl so in-flight code in this JS execution sees the new value
 * immediately.
 */
function isValidHttpUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

export function switchToWorkerUrl(workerUrl: string | null): void {
  if (workerUrl && !isValidHttpUrl(workerUrl)) {
    return;
  }
  if (typeof window !== "undefined") {
    if (workerUrl) {
      window.localStorage.setItem(WORKER_URL_KEY, workerUrl);
    } else {
      window.localStorage.removeItem(WORKER_URL_KEY);
    }
  }
  proxyBaseUrl = workerUrl ?? defaultProxyBaseUrl;
}

const HTTP_REQUEST = {
  GET: "GET",
  POST: "POST",
  PUT: "PUT",
  DELETE: "DELETE",
};

export interface Model {
  model_name: string;
  litellm_params: object;
  model_info: object | null;
}

interface PromptInfo {
  prompt_type: string;
  environment?: string;
}

export interface PromptSpec {
  prompt_id: string;
  litellm_params: object;
  prompt_info: PromptInfo;
  created_at?: string;
  updated_at?: string;
  version?: number; // Explicit version number for version history
  environment?: string;
  created_by?: string;
}

export interface PromptTemplateBase {
  litellm_prompt_id: string;
  content: string;
  metadata?: Record<string, unknown> | null;
}

interface PromptInfoResponse {
  prompt_spec: PromptSpec;
  raw_prompt_template: PromptTemplateBase | null;
  environments?: string[];
}

export interface ListPromptsResponse {
  prompts: PromptSpec[];
}

export interface Organization {
  organization_id: string;
  organization_alias: string;
  budget_id: string;
  metadata: Record<string, any>;
  models: string[];
  spend: number;
  model_spend: Record<string, number>;
  created_at: string;
  created_by: string;
  updated_at: string;
  updated_by: string;
  litellm_budget_table: any; // Simplified to any since we don't need the detailed structure
  teams: any[] | null;
  users: any[] | null;
  members: any[] | null;
  object_permission?: {
    object_permission_id: string;
    mcp_servers: string[];
    mcp_access_groups?: string[];
    vector_stores: string[];
  };
}

export interface CredentialItem {
  credential_name: string;
  credential_values: any;
  credential_info: {
    custom_llm_provider?: string;
    description?: string;
    required?: boolean;
  };
}

export interface ProviderCredentialFieldMetadata {
  key: string;
  label: string;
  placeholder?: string | null;
  tooltip?: string | null;
  required?: boolean;
  field_type?: "text" | "password" | "select" | "upload" | "textarea";
  options?: string[] | null;
  default_value?: string | null;
}

export interface ProviderCreateInfo {
  provider: string;
  provider_display_name: string;
  litellm_provider: string;
  default_model_placeholder?: string | null;
  credential_fields: ProviderCredentialFieldMetadata[];
}

export interface AgentCredentialFieldMetadata {
  key: string;
  label: string;
  placeholder?: string | null;
  tooltip?: string | null;
  required?: boolean;
  field_type?: "text" | "password" | "select" | "upload" | "textarea";
  options?: string[] | null;
  default_value?: string | null;
  include_in_litellm_params?: boolean;
}

export interface AgentCreateInfo {
  agent_type: string;
  agent_type_display_name: string;
  description?: string | null;
  logo_url?: string | null;
  credential_fields: AgentCredentialFieldMetadata[];
  litellm_params_template?: Record<string, string> | null;
  model_template?: string | null;
  use_a2a_form_fields?: boolean;
}

interface PublicModelHubInfo {
  docs_title: string;
  custom_docs_description: string | null;
  litellm_version: string;
  // Supports both old format (Record<string, string>) and new format (Record<string, {url: string, index: number}>)
  useful_links: Record<string, string | { url: string; index: number }>;
}

export interface WorkerInfo {
  worker_id: string;
  name: string;
  url: string;
}

export interface LiteLLMWellKnownUiConfig {
  server_root_path: string;
  proxy_base_url: string | null;
  auto_redirect_to_sso: boolean;
  admin_ui_disabled: boolean;
  sso_configured: boolean;
  hide_default_credentials_hint?: boolean;
  is_control_plane?: boolean;
  workers?: WorkerInfo[];
}

export interface CredentialsResponse {
  credentials: CredentialItem[];
}

let lastErrorTime = 0;

export const handleError = async (errorData: string | any) => {
  const currentTime = Date.now();
  if (currentTime - lastErrorTime > 60000) {
    // 60000 milliseconds = 60 seconds
    // Convert errorData to string if it isn't already
    const errorString = typeof errorData === "string" ? errorData : JSON.stringify(errorData);
    if (errorString.includes("Authentication Error - Expired Key")) {
      NotificationsManager.info("UI Session Expired. Logging out.");
      lastErrorTime = currentTime;
      clearTokenCookies();
      const browserLocation = getWindowLocation();
      if (browserLocation) {
        window.location.href = browserLocation.pathname;
      }
    }
    lastErrorTime = currentTime;
  }
};

export const getProviderCreateMetadata = async (): Promise<ProviderCreateInfo[]> => {
  /**
   * Fetch provider credential field metadata from the proxy's public endpoint.
   * This is used by the UI to dynamically render provider-specific credential fields.
   */
  const url = proxyBaseUrl ? `${proxyBaseUrl}/public/providers/fields` : `/public/providers/fields`;
  const response = await fetch(url, {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Failed to fetch provider create metadata:", response.status, errorText);
    throw new Error("Failed to load provider configuration");
  }

  const jsonData: ProviderCreateInfo[] = await response.json();
  return jsonData;
};

export const getAgentCreateMetadata = async (): Promise<AgentCreateInfo[]> => {
  /**
   * Fetch agent type metadata from the proxy's public endpoint.
   * This is used by the UI to dynamically render agent-specific credential fields.
   */
  const url = proxyBaseUrl ? `${proxyBaseUrl}/public/agents/fields` : `/public/agents/fields`;
  const response = await fetch(url, {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Failed to fetch agent create metadata:", response.status, errorText);
    throw new Error("Failed to load agent configuration");
  }

  const jsonData: AgentCreateInfo[] = await response.json();
  return jsonData;
};

// Global variable for the header name
let globalLitellmHeaderName: string = "Authorization";
const MCP_AUTH_HEADER: string = "x-mcp-auth";

// Function to set the global header name
export function setGlobalLitellmHeaderName(headerName: string = "Authorization") {
  globalLitellmHeaderName = headerName;
}

// Function to get the global header name
export function getGlobalLitellmHeaderName(): string {
  return globalLitellmHeaderName;
}

export const apiClient = createApiClient({
  getBaseUrl: getProxyBaseUrl,
  getAuthHeaderName: getGlobalLitellmHeaderName,
  onError: handleError,
});

registerBaseUrlGetter(getProxyBaseUrl);
registerAuthHeaderNameGetter(getGlobalLitellmHeaderName);
registerAuthTokenGetter(() => decodeToken(getCookie("token"))?.key ?? null);
registerErrorHandler(handleError);

export const makeModelGroupPublic = async (accessToken: string, modelGroups: string[]) => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/model_group/make_public` : `/model_group/make_public`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model_groups: modelGroups,
    }),
  });
  return response.json();
};

export const getUiConfig = async () => {
  /**Special route to get the proxy base url and server root path */
  const url = defaultProxyBaseUrl
    ? `${defaultProxyBaseUrl}/litellm/.well-known/litellm-ui-config`
    : `/litellm/.well-known/litellm-ui-config`;
  const response = await fetch(url);
  const jsonData: LiteLLMWellKnownUiConfig = await response.json();
  /**
   * Update the proxy base url and server root path
   */
  updateServerRootPath(jsonData.server_root_path);
  updateProxyBaseUrl(jsonData.server_root_path, jsonData.proxy_base_url);
  return jsonData;
};

export const getPublicModelHubInfo = async () => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/public/model_hub/info` : `/public/model_hub/info`;
  const response = await fetch(url);
  const jsonData: PublicModelHubInfo = await response.json();
  return jsonData;
};

export const getOpenAPISchema = async () => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/openapi.json` : `/openapi.json`;
  const response = await fetch(url);
  const jsonData = await response.json();
  return jsonData;
};

export const modelCostMap = async () => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/public/litellm_model_cost_map` : `/public/litellm_model_cost_map`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });
    const jsonData = await response.json();
    return jsonData;
  } catch (error) {
    console.error("Failed to get model cost map:", error);
    throw error;
  }
};

export const reloadModelCostMap = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/reload/model_cost_map` : `/reload/model_cost_map`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    const jsonData = await response.json();
    return jsonData;
  } catch (error) {
    console.error("Failed to reload model cost map:", error);
    throw error;
  }
};

export const scheduleModelCostMapReload = async (accessToken: string, hours: number) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/schedule/model_cost_map_reload?hours=${hours}`
      : `/schedule/model_cost_map_reload?hours=${hours}`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    const jsonData = await response.json();
    return jsonData;
  } catch (error) {
    console.error("Failed to schedule model cost map reload:", error);
    throw error;
  }
};

export const cancelModelCostMapReload = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/schedule/model_cost_map_reload` : `/schedule/model_cost_map_reload`;
    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    const jsonData = await response.json();
    return jsonData;
  } catch (error) {
    console.error("Failed to cancel model cost map reload:", error);
    throw error;
  }
};

export const getModelCostMapSource = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/model/cost_map/source` : `/model/cost_map/source`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const jsonData = await response.json();
    return jsonData;
  } catch (error) {
    console.error("Failed to get model cost map source info:", error);
    throw error;
  }
};

export const getModelCostMapReloadStatus = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/schedule/model_cost_map_reload/status`
      : `/schedule/model_cost_map_reload/status`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      console.error(`Status request failed with status: ${response.status}`);
      const errorText = await response.text();
      console.error("Error response:", errorText);
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const jsonData = await response.json();
    return jsonData;
  } catch (error) {
    console.error("Failed to get model cost map reload status:", error);
    throw error;
  }
};
export const modelCreateCall = async (accessToken: string, formValues: Model) => {
  try {
    const data = await apiClient.post(`/model/new`, {
      accessToken,
      body: {
        ...formValues,
      },
    });

    // Close any existing messages before showing new ones
    MessageManager.destroy();

    // Sequential success messages
    NotificationsManager.success(`Model ${formValues.model_name} created successfully`);

    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelDeleteCall = async (accessToken: string, model_id: string) => {
  try {
    const data = await apiClient.post(`/model/delete`, {
      accessToken,
      body: {
        id: model_id,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const budgetDeleteCall = async (accessToken: string | null, budget_id: string) => {
  if (accessToken == null) {
    return;
  }

  try {
    const data = await apiClient.post(`/budget/delete`, {
      accessToken,
      body: {
        id: budget_id,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const budgetCreateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    const data = await apiClient.post(`/budget/new`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const budgetUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    const data = await apiClient.post(`/budget/update`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const invitationCreateCall = async (
  accessToken: string,
  userID: string, // Assuming formValues is an object
) => {
  try {
    const data = await apiClient.post(`/invitation/new`, {
      accessToken,
      body: {
        user_id: userID, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const alertingSettingsCall = async (accessToken: string) => {
  /**
   * Get all configurable params for setting a model
   */
  try {
    return await apiClient.get(`/alerting/settings`, { accessToken });
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const keyCreateServiceAccountCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    // check if formValues.description is not undefined, make it a string and add it to formValues.metadata
    if (formValues.description) {
      // add to formValues.metadata
      if (!formValues.metadata) {
        formValues.metadata = {};
      }
      // value needs to be in "", valid JSON
      formValues.metadata.description = formValues.description;
      // remove descrption from formValues
      delete formValues.description;
      formValues.metadata = JSON.stringify(formValues.metadata);
    }
    // Parse JSON fields if they exist
    for (const field of jsonFields) {
      if (formValues[field]) {
        // if there's an exception JSON.parse, show it in the message
        try {
          formValues[field] = JSON.parse(formValues[field]);
        } catch (error) {
          throw new Error(`Failed to parse ${field}: ` + error);
        }
      }
    }

    const url = proxyBaseUrl ? `${proxyBaseUrl}/key/service-account/generate` : `/key/service-account/generate`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keyCreateCall = async (
  accessToken: string,
  userID: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    // check if formValues.description is not undefined, make it a string and add it to formValues.metadata
    if (formValues.description) {
      // add to formValues.metadat
      if (!formValues.metadata) {
        formValues.metadata = {};
      }
      // value needs to be in "", valid JSON
      formValues.metadata.description = formValues.description;
      // remove descrption from formValues
      delete formValues.description;
      formValues.metadata = JSON.stringify(formValues.metadata);
    }
    // Parse JSON fields if they exist
    for (const field of jsonFields) {
      if (formValues[field]) {
        // if there's an exception JSON.parse, show it in the message
        try {
          formValues[field] = JSON.parse(formValues[field]);
        } catch (error) {
          throw new Error(`Failed to parse ${field}: ` + error);
        }
      }
    }

    const url = proxyBaseUrl ? `${proxyBaseUrl}/key/generate` : `/key/generate`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userID,
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keyCreateForAgentCall = async (
  accessToken: string,
  agentId: string,
  keyAlias: string,
  models: string[],
  metadata?: Record<string, any>,
  teamId?: string | null,
) => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/key/generate` : `/key/generate`;
  const body: Record<string, any> = {
    agent_id: agentId,
    key_alias: keyAlias,
    models: models.length > 0 ? models : [],
  };
  if (teamId) {
    body.team_id = teamId;
  }
  if (metadata && Object.keys(metadata).length > 0) {
    body.metadata = metadata;
  }
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error("Failed to create key for agent");
  }

  return response.json();
};

export const userCreateCall = async (
  accessToken: string,
  userID: string | null,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    // check if formValues.description is not undefined, make it a string and add it to formValues.metadata
    if (formValues.description) {
      // add to formValues.metadata
      if (!formValues.metadata) {
        formValues.metadata = {};
      }
      // value needs to be in "", valid JSON
      formValues.metadata.description = formValues.description;
      // remove descrption from formValues
      delete formValues.description;
      formValues.metadata = JSON.stringify(formValues.metadata);
    }

    formValues.auto_create_key = false;
    // if formValues.metadata is not undefined, make it a valid dict
    if (formValues.metadata) {
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/new` : `/user/new`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userID,
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keyDeleteCall = async (accessToken: string, user_key: string) => {
  try {
    return await apiClient.post(`/key/delete`, { accessToken, body: { keys: [user_key] } });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export interface KeyShareResponse {
  share_link: string;
}

export const keyShareCreateCall = async (accessToken: string, key: string): Promise<KeyShareResponse> => {
  try {
    return await apiClient.post<KeyShareResponse>(`/key/share`, { accessToken, body: { key } });
  } catch (error) {
    console.error("Failed to create secure share link:", error);
    throw error;
  }
};

export const userDeleteCall = async (accessToken: string, userIds: string[]) => {
  try {
    return await apiClient.post(`/user/delete`, { accessToken, body: { user_ids: userIds } });
  } catch (error) {
    console.error("Failed to delete user(s):", error);
    throw error;
  }
};

export const teamDeleteCall = async (accessToken: string, teamID: string) => {
  try {
    return await apiClient.post(`/team/delete`, { accessToken, body: { team_ids: [teamID] } });
  } catch (error) {
    console.error("Failed to delete key:", error);
    throw error;
  }
};

export interface UserInfo {
  user_id: string;
  user_email: string;
  user_alias: string | null;
  user_role: string;
  spend: number;
  max_budget: number | null;
  models: string[];
  key_count: number;
  created_at: string;
  updated_at: string;
  sso_user_id: string | null;
  budget_duration: string | null;
  metadata?: Record<string, unknown> | null;
}

export type UserListResponse = {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  users: UserInfo[];
};

export const userListCall = async (
  accessToken: string,
  userIDs: string[] | null = null,
  page: number | null = null,
  page_size: number | null = null,
  userEmail: string | null = null,
  userRole: string | null = null,
  team: string | null = null,
  sso_user_id: string | null = null,
  sortBy: string | null = null,
  sortOrder: "asc" | "desc" | null = null,
  organizationIds: string[] | null = null,
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    const data = (await apiClient.get(`/user/list`, {
      accessToken,
      query: {
        user_ids: userIDs && userIDs.length > 0 ? userIDs.join(",") : undefined,
        page: page || undefined,
        page_size: page_size || undefined,
        user_email: userEmail || undefined,
        role: userRole || undefined,
        team: team || undefined,
        sso_user_ids: sso_user_id || undefined,
        sort_by: sortBy || undefined,
        sort_order: sortOrder || undefined,
        organization_ids: organizationIds && organizationIds.length > 0 ? organizationIds.join(",") : undefined,
      },
    })) as UserListResponse;
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

/**
 * Response type for /v2/user/info — lightweight endpoint that returns only the user object.
 */
export interface UserInfoV2Response {
  user_id: string;
  user_email: string | null;
  user_alias: string | null;
  user_role: string | null;
  spend: number;
  max_budget: number | null;
  models: string[];
  budget_duration: string | null;
  budget_reset_at: string | null;
  metadata: Record<string, any> | null;
  created_at: string | null;
  updated_at: string | null;
  sso_user_id: string | null;
  teams: string[];
}

/**
 * Lightweight user info fetch from /v2/user/info.
 * Returns only the user object — no keys, no teams objects.
 *
 * @param accessToken - Bearer token for auth
 * @param userId - Optional user ID to look up. If omitted, returns the caller's own info.
 */
export const userGetInfoV2 = async (accessToken: string, userId?: string): Promise<UserInfoV2Response> => {
  try {
    return await apiClient.get(`/v2/user/info`, { accessToken, query: { user_id: userId || undefined } });
  } catch (error) {
    console.error("Failed to fetch user info v2:", error);
    throw error;
  }
};

export const userInfoCall = async (
  accessToken: string,
  userID: string | null,
  userRole: string,
  viewAll: boolean = false,
  page: number | null,
  page_size: number | null,
  lookup_user_id: boolean = false,
) => {
  try {
    if (viewAll) {
      return await apiClient.get(`/user/list`, {
        accessToken,
        query: {
          page: page != null ? page.toString() : undefined,
          page_size: page_size != null ? page_size.toString() : undefined,
        },
      });
    }

    const includeUserID = !((userRole === "Admin" || userRole === "Admin Viewer") && !lookup_user_id) && userID;
    return await apiClient.get(`/user/info`, {
      accessToken,
      query: { user_id: includeUserID ? userID : undefined },
    });
  } catch (error) {
    console.error("Failed to fetch user data:", error);
    throw error;
  }
};

export const teamInfoCall = async (accessToken: string, teamID: string | null) => {
  try {
    return await apiClient.get(`/team/info`, { accessToken, query: { team_id: teamID || undefined } });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

type TeamListResponse = {
  teams: Team[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
};

export const v2TeamListCall = async (
  accessToken: string,
  organizationID: string | null,
  userID: string | null = null,
  teamID: string | null = null,
  team_alias: string | null = null,
  page: number = 1,
  page_size: number = 10,
  sort_by: string | null = null,
  sort_order: "asc" | "desc" | null = null,
): Promise<TeamListResponse> => {
  /**
   * Get list of teams with filtering and sorting options
   */
  try {
    return await apiClient.get(`/v2/team/list`, {
      accessToken,
      query: {
        user_id: userID || undefined,
        organization_id: organizationID || undefined,
        team_id: teamID || undefined,
        team_alias: team_alias || undefined,
      },
    });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const teamListCall = async (
  accessToken: string,
  organizationID: string | null,
  userID: string | null = null,
  teamID: string | null = null,
  team_alias: string | null = null,
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    return await apiClient.get(`/team/list`, {
      accessToken,
      query: {
        user_id: userID || undefined,
        organization_id: organizationID || undefined,
        team_id: teamID || undefined,
        team_alias: team_alias || undefined,
      },
    });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const availableTeamListCall = async (accessToken: string) => {
  /**
   * Get all available teams on proxy
   */
  try {
    const data = await apiClient.get(`/team/available`, { accessToken });
    return data;
  } catch (error) {
    throw error;
  }
};

export const organizationListCall = async (
  accessToken: string,
  org_id: string | null = null,
  org_alias: string | null = null,
) => {
  /**
   * Get all organizations on proxy
   */
  try {
    return await apiClient.get(`/organization/list`, {
      accessToken,
      query: {
        org_id: org_id || undefined,
        org_alias: org_alias || undefined,
      },
    });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const organizationInfoCall = async (accessToken: string, organizationID: string) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/organization/info` : `/organization/info`;
    if (organizationID) {
      url = `${url}?organization_id=${organizationID}`;
    }
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const organizationCreateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    if (formValues.metadata) {
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        console.error("Failed to parse metadata:", error);
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const data = await apiClient.post(`/organization/new`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const organizationUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    const data = await apiClient.patch(`/organization/update`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const organizationDeleteCall = async (accessToken: string, organizationID: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/organization/delete` : `/organization/delete`;
    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        organization_ids: [organizationID],
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(`Error deleting organization: ${errorData}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to delete organization:", error);
    throw error;
  }
};

export const transformRequestCall = async (accessToken: string, request: object) => {
  /**
   * Transform request
   */

  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/utils/transform_request` : `/utils/transform_request`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

type DailyActivityQueryValue = string | number | string[] | null | undefined;

const DEFAULT_DAILY_ACTIVITY_PAGE_SIZE = "1000";

const appendDailyActivityQueryParam = (params: URLSearchParams, key: string, value: DailyActivityQueryValue) => {
  if (value === null || value === undefined) {
    return;
  }

  if (Array.isArray(value)) {
    if (value.length > 0) {
      params.append(key, value.join(","));
    }
    return;
  }

  params.append(key, `${value}`);
};

const buildDailyActivityUrl = (
  endpoint: string,
  startTime: Date,
  endTime: Date,
  page: number,
  extraQueryParams?: Record<string, DailyActivityQueryValue>,
) => {
  const resolvedEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  const baseUrl = proxyBaseUrl ? `${proxyBaseUrl}${resolvedEndpoint}` : resolvedEndpoint;

  const params = new URLSearchParams();
  params.append("start_date", formatDate(startTime));
  params.append("end_date", formatDate(endTime));
  params.append("page_size", DEFAULT_DAILY_ACTIVITY_PAGE_SIZE);
  params.append("page", page.toString());
  // Send timezone offset so backend can adjust date range for UTC storage
  params.append("timezone", new Date().getTimezoneOffset().toString());

  if (extraQueryParams) {
    Object.entries(extraQueryParams).forEach(([key, value]) => {
      appendDailyActivityQueryParam(params, key, value);
    });
  }

  const queryString = params.toString();
  return queryString ? `${baseUrl}?${queryString}` : baseUrl;
};

type DailyActivityCallOptions = {
  accessToken: string;
  endpoint: string;
  startTime: Date;
  endTime: Date;
  page?: number;
  extraQueryParams?: Record<string, DailyActivityQueryValue>;
};

const fetchDailyActivity = async ({
  accessToken,
  endpoint,
  startTime,
  endTime,
  page = 1,
  extraQueryParams,
}: DailyActivityCallOptions) => {
  try {
    const url = buildDailyActivityUrl(endpoint, startTime, endTime, page, extraQueryParams);

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to fetch daily activity (${endpoint}):`, error);
    throw error;
  }
};

export const userDailyActivityCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  userId: string | null = null,
) => {
  /**
   * Get daily user activity on proxy
   */
  return fetchDailyActivity({
    accessToken,
    endpoint: "/user/daily/activity",
    startTime,
    endTime,
    page,
    extraQueryParams: {
      user_id: userId,
    },
  });
};

export const tagDailyActivityCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  tags: string[] | null = null,
) => {
  /**
   * Get daily user activity on proxy
   */
  return fetchDailyActivity({
    accessToken,
    endpoint: "/tag/daily/activity",
    startTime,
    endTime,
    page,
    extraQueryParams: {
      tags,
    },
  });
};

export const teamDailyActivityCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  teamIds: string[] | null = null,
) => {
  /**
   * Get daily user activity on proxy
   */
  return fetchDailyActivity({
    accessToken,
    endpoint: "/team/daily/activity",
    startTime,
    endTime,
    page,
    extraQueryParams: {
      team_ids: teamIds,
      exclude_team_ids: "litellm-dashboard",
    },
  });
};

export const organizationDailyActivityCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  organizationIds: string[] | null = null,
) => {
  return fetchDailyActivity({
    accessToken,
    endpoint: "/organization/daily/activity",
    startTime,
    endTime,
    page,
    extraQueryParams: {
      organization_ids: organizationIds,
    },
  });
};

export const customerDailyActivityCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  customerIds: string[] | null = null,
) => {
  return fetchDailyActivity({
    accessToken,
    endpoint: "/customer/daily/activity",
    startTime,
    endTime,
    page,
    extraQueryParams: {
      end_user_ids: customerIds,
    },
  });
};

export const agentDailyActivityCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  agentIds: string[] | null = null,
) => {
  return fetchDailyActivity({
    accessToken,
    endpoint: "/agent/daily/activity",
    startTime,
    endTime,
    page,
    extraQueryParams: {
      agent_ids: agentIds,
    },
  });
};

export const getOnboardingCredentials = async (inviteUUID: string) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/onboarding/get_token` : `/onboarding/get_token`;
    url += `?invite_link=${inviteUUID}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const claimOnboardingToken = async (
  accessToken: string,
  inviteUUID: string,
  userID: string,
  password: string,
) => {
  try {
    const data = await apiClient.post(`/onboarding/claim_token`, {
      accessToken,
      body: {
        invitation_link: inviteUUID,
        user_id: userID,
        password: password,
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to delete key:", error);
    throw error;
  }
};

export const regenerateKeyCall = async (accessToken: string, keyToRegenerate: string, formData: any) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/key/${keyToRegenerate}/regenerate`
      : `/key/${keyToRegenerate}/regenerate`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to regenerate key:", error);
    throw error;
  }
};

let ModelListerrorShown = false;
let errorTimer: NodeJS.Timeout | null = null;

export const modelInfoCall = async (
  accessToken: string,
  userID: string,
  userRole: string,
  page: number = 1,
  size: number = 50,
  search?: string,
  modelId?: string,
  teamId?: string,
  sortBy?: string,
  sortOrder?: string,
) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/v2/model/info` : `/v2/model/info`;
    const params = new URLSearchParams();
    params.append("include_team_models", "true");
    params.append("page", page.toString());
    params.append("size", size.toString());
    if (search && search.trim()) {
      params.append("search", search.trim());
    }
    if (modelId && modelId.trim()) {
      params.append("modelId", modelId.trim());
    }
    if (teamId && teamId.trim()) {
      params.append("teamId", teamId.trim());
    }
    if (sortBy && sortBy.trim()) {
      params.append("sortBy", sortBy.trim());
    }
    if (sortOrder && sortOrder.trim()) {
      params.append("sortOrder", sortOrder.trim());
    }
    if (params.toString()) {
      url += `?${params.toString()}`;
    }

    //NotificationsManager.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      let errorData = await response.text();
      errorData += `error shown=${ModelListerrorShown}`;
      if (!ModelListerrorShown) {
        if (errorData.includes("No model list passed")) {
          errorData = "No Models Exist. Click Add Model to get started.";
        }
        NotificationsManager.info(errorData);
        ModelListerrorShown = true;

        if (errorTimer) clearTimeout(errorTimer);
        errorTimer = setTimeout(() => {
          ModelListerrorShown = false;
        }, 10000);
      }

      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelInfoV1Call = async (accessToken: string, modelId: string) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/v1/model/info` : `/v1/model/info`;
    url += `?litellm_model_id=${modelId}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelHubPublicModelsCall = async () => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/public/model_hub` : `/public/model_hub`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    console.error(`modelHubPublicModelsCall failed with status ${response.status}`);
    return [];
  }
  return response.json();
};

export const agentHubPublicModelsCall = async () => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/public/agent_hub` : `/public/agent_hub`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    console.error(`agentHubPublicModelsCall failed with status ${response.status}`);
    return [];
  }
  return response.json();
};

export const mcpHubPublicServersCall = async () => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/public/mcp_hub` : `/public/mcp_hub`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    console.error(`mcpHubPublicServersCall failed with status ${response.status}`);
    return [];
  }
  return response.json();
};

export const skillHubPublicCall = async () => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/public/skill_hub` : `/public/skill_hub`;
  const response = await fetch(url, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    console.error(`skillHubPublicCall failed with status ${response.status}`);
    return { plugins: [] };
  }
  return response.json();
};

export const modelHubCall = async (accessToken: string) => {
  /**
   * Get all models on proxy
   */
  try {
    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.get(`/model_group/info`, { accessToken });
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

// Function to get allowed IPs
export const getAllowedIPs = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/get/allowed_ips`, { accessToken });
    return data.data; // Assuming the API returns { data: [...] }
  } catch (error) {
    console.error("Failed to get allowed IPs:", error);
    throw error;
  }
};

// Function to add an allowed IP
export const addAllowedIP = async (accessToken: string, ip: string) => {
  try {
    const data = await apiClient.post(`/add/allowed_ip`, { accessToken, body: { ip: ip } });
    return data;
  } catch (error) {
    console.error("Failed to add allowed IP:", error);
    throw error;
  }
};

// Function to delete an allowed IP
export const deleteAllowedIP = async (accessToken: string, ip: string) => {
  try {
    const data = await apiClient.post(`/delete/allowed_ip`, { accessToken, body: { ip: ip } });
    return data;
  } catch (error) {
    console.error("Failed to delete allowed IP:", error);
    throw error;
  }
};

export const updateUsefulLinksCall = async (
  accessToken: string,
  useful_links: Record<string, string | { url: string; index: number }>,
) => {
  try {
    return await apiClient.post(`/model_hub/update_useful_links`, {
      accessToken,
      body: { useful_links: useful_links },
    });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelAvailableCall = async (
  accessToken: string,
  userID: string,
  userRole: string,
  return_wildcard_routes: boolean = false,
  teamID: string | null = null,
  include_model_access_groups: boolean = false,
  only_model_access_groups: boolean = false,
  scope?: string,
) => {
  /**
   * Get all the models user has access to
   */
  try {
    return await apiClient.get(`/models`, {
      accessToken,
      query: {
        include_model_access_groups: "True",
        return_wildcard_routes: return_wildcard_routes === true ? "True" : undefined,
        only_model_access_groups: only_model_access_groups === true ? "True" : undefined,
        team_id: teamID || undefined,
        scope: scope || undefined,
      },
    });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const teamSpendLogsCall = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/global/spend/teams`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const tagsSpendLogsCall = async (
  accessToken: string,
  startTime: string | undefined,
  endTime: string | undefined,
  tags: string[] | undefined,
) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/global/spend/tags` : `/global/spend/tags`;

    if (startTime && endTime) {
      url = `${url}?start_date=${startTime}&end_date=${endTime}`;
    }

    // if tags, convert the list to a comma separated string
    if (tags) {
      url += `&tags=${tags.join(",")}`;
    }

    const response = await fetch(`${url}`, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const allTagNamesCall = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/global/spend/all_tag_names`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const allEndUsersCall = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/customer/list`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to fetch end users:", error);
    throw error;
  }
};

export const userFilterUICall = async (accessToken: string, params: URLSearchParams) => {
  try {
    return await apiClient.get(`/user/filter/ui`, {
      accessToken,
      query: {
        user_email: params.get("user_email") || undefined,
        user_id: params.get("user_id") || undefined,
        team_id: params.get("team_id") || undefined,
      },
    });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

/**
 * Optional query params for /spend/logs/ui - matches backend spend_management_endpoints.py
 */
interface UiSpendLogsParams {
  api_key?: string;
  team_id?: string;
  request_id?: string;
  session_id?: string;
  user_id?: string;
  end_user?: string;
  status_filter?: string;
  /** Filter by model name (e.g. "gpt-4") */
  model?: string;
  /** Filter by model ID (litellm model deployment id) */
  model_id?: string;
  key_alias?: string;
  error_code?: string;
  error_message?: string;
  sort_by?: string;
  sort_order?: "asc" | "desc";
  min_spend?: number;
  max_spend?: number;
}

interface UiSpendLogsCallOptions {
  accessToken: string;
  start_date: string;
  end_date: string;
  page?: number;
  page_size?: number;
  params?: UiSpendLogsParams;
}

export const uiSpendLogsCall = async ({
  accessToken,
  start_date,
  end_date,
  page = 1,
  page_size = 50,
  params = {},
}: UiSpendLogsCallOptions) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl ? `${proxyBaseUrl}/spend/logs/ui` : `/spend/logs/ui`;

    const queryParams = new URLSearchParams();
    queryParams.append("start_date", start_date);
    queryParams.append("end_date", end_date);
    queryParams.append("page", page.toString());
    queryParams.append("page_size", page_size.toString());

    // Add optional params only when explicitly provided
    for (const [key, value] of Object.entries(params)) {
      if (value == null) continue;
      if (key === "min_spend" || key === "max_spend") {
        queryParams.append(key, value.toString());
      } else if (typeof value === "string" && value !== "") {
        queryParams.append(key, String(value));
      }
    }

    const queryString = queryParams.toString();
    if (queryString) {
      url += `?${queryString}`;
    }

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch spend logs:", error);
    throw error;
  }
};

export const adminSpendLogsCall = async (accessToken: string) => {
  try {
    //NotificationsManager.info("Making spend logs request");
    const data = await apiClient.get(`/global/spend/logs`, { accessToken });
    //NotificationsManager.success("Spend Logs received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const adminTopKeysCall = async (accessToken: string) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/global/spend/keys?limit=5` : `/global/spend/keys?limit=5`;

    //NotificationsManager.info("Making spend keys request");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    //NotificationsManager.success("Spend Logs received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const adminTopEndUsersCall = async (
  accessToken: string,
  keyToken: string | null,
  startTime: string | undefined,
  endTime: string | undefined,
) => {
  try {
    const body = keyToken
      ? { api_key: keyToken, startTime: startTime, endTime: endTime }
      : { startTime: startTime, endTime: endTime };

    //NotificationsManager.info("Making top end users request");
    const data = await apiClient.post(`/global/spend/end_users`, { accessToken, body });
    //NotificationsManager.success("Top End users received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const adminspendByProvider = async (
  accessToken: string,
  keyToken: string | null,
  startTime: string | undefined,
  endTime: string | undefined,
) => {
  try {
    const data = await apiClient.get(`/global/spend/provider`, {
      accessToken,
      query: {
        ...(startTime && endTime ? { start_date: startTime, end_date: endTime } : {}),
        ...(keyToken ? { api_key: keyToken } : {}),
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminGlobalActivity = async (
  accessToken: string,
  startTime: string | undefined,
  endTime: string | undefined,
) => {
  try {
    const data = await apiClient.get(`/global/activity`, {
      accessToken,
      query: startTime && endTime ? { start_date: startTime, end_date: endTime } : undefined,
    });
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminGlobalCacheActivity = async (
  accessToken: string,
  startTime: string | undefined,
  endTime: string | undefined,
) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/global/activity/cache_hits` : `/global/activity/cache_hits`;

    if (startTime && endTime) {
      url += `?start_date=${startTime}&end_date=${endTime}`;
    }

    const requestOptions = {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    };

    const response = await fetch(url, requestOptions);

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminGlobalActivityPerModel = async (
  accessToken: string,
  startTime: string | undefined,
  endTime: string | undefined,
) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/global/activity/model` : `/global/activity/model`;

    if (startTime && endTime) {
      url += `?start_date=${startTime}&end_date=${endTime}`;
    }

    const requestOptions = {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    };

    const response = await fetch(url, requestOptions);

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminTopModelsCall = async (accessToken: string) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/global/spend/models?limit=5` : `/global/spend/models?limit=5`;

    //NotificationsManager.info("Making top models request");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    //NotificationsManager.success("Top Models received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keyInfoCall = async (accessToken: string, keys: string[]) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/v2/key/info` : `/v2/key/info`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        keys: keys,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      if (errorData.includes("Invalid proxy server token passed")) {
        throw new Error("Invalid proxy server token passed");
      }
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const testConnectionRequest = async (
  accessToken: string,
  litellm_params: Record<string, any>,
  model_info: Record<string, any>,
  mode: string,
) => {
  try {
    // Construct the URL based on environment
    const url = proxyBaseUrl ? `${proxyBaseUrl}/health/test_connection` : `/health/test_connection`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({
        litellm_params: litellm_params,
        model_info: model_info,
        mode: mode,
      }),
    });

    // Check for non-JSON responses first
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const text = await response.text();
      console.error("Received non-JSON response:", text);
      throw new Error(
        `Received non-JSON response (${response.status}: ${response.statusText}). Check network tab for details.`,
      );
    }

    const data = await response.json();

    if (!response.ok || data.status === "error") {
      // Return the error response instead of throwing an error
      // This allows the caller to handle the error format properly
      if (data.status === "error") {
        return data; // Return the full error response
      } else {
        return {
          status: "error",
          message: data.error?.message || `Connection test failed: ${response.status} ${response.statusText}`,
        };
      }
    }

    return data;
  } catch (error) {
    console.error("Model connection test error:", error);
    // For network errors or other exceptions, still throw
    throw error;
  }
};

export type ModelGroupConnectionResult = { status: "success" } | { status: "error"; error: string };

/**
 * Test an existing model group by routing a minimal request through the proxy
 * exactly as production would (by public model_group name). Unlike
 * /health/test_connection, this needs no litellm_params resolution: the router
 * resolves the group, credentials, and provider. Used by the auto-router Test
 * Connection to probe each tier's model group and the embedding model.
 */
/**
 * Build the minimal request that probes a model group by public name. No
 * max_tokens: reasoning models (o1/o3/...) reject a tiny cap with "max_tokens
 * reached" because reasoning tokens count against it, which would show a false
 * failure for a reachable tier.
 */
export const buildModelGroupTestRequest = (
  modelGroup: string,
  mode: "chat" | "embedding",
): { path: string; body: Record<string, unknown> } =>
  mode === "embedding"
    ? { path: "/v1/embeddings", body: { model: modelGroup, input: "test from litellm" } }
    : {
        path: "/v1/chat/completions",
        body: { model: modelGroup, messages: [{ role: "user", content: "test from litellm" }] },
      };

export const testModelGroupConnection = async (
  accessToken: string,
  modelGroup: string,
  mode: "chat" | "embedding",
): Promise<ModelGroupConnectionResult> => {
  const { path, body } = buildModelGroupTestRequest(modelGroup, mode);
  try {
    await apiClient.post(path, { accessToken, body });
    return { status: "success" };
  } catch (error) {
    return { status: "error", error: error instanceof Error ? error.message : String(error) };
  }
};

// ... existing code ...
export const keyInfoV1Call = async (accessToken: string, key: string) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/key/info` : `/key/info`;
    url = `${url}?key=${key}`; // Add key as query parameter

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      // Remove body since this is a GET request
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      NotificationsManager.fromBackend("Failed to fetch key info - " + errorData);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch key info:", error);
    throw error;
  }
};

export const keyListCall = async (
  accessToken: string,
  organizationID: string | null,
  teamID: string | null,
  selectedKeyAlias: string | null,
  userID: string | null,
  keyHash: string | null,
  page: number,
  pageSize: number,
  sortBy: string | null = null,
  sortOrder: string | null = null,
  expand: string | null = null,
  status: string | null = null,
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    return await apiClient.get(`/key/list`, {
      accessToken,
      query: {
        team_id: teamID || undefined,
        organization_id: organizationID || undefined,
        key_alias: selectedKeyAlias || undefined,
        key_hash: keyHash || undefined,
        user_id: userID || undefined,
        page: page ? page.toString() : undefined,
        size: pageSize ? pageSize.toString() : undefined,
        sort_by: sortBy || undefined,
        sort_order: sortOrder || undefined,
        expand: expand || undefined,
        status: status || undefined,
        return_full_object: "true",
        include_team_keys: "true",
        include_created_by_keys: "true",
        // /key/list is exact by default; opt in so the key-list search box keeps
        // matching partial user_id/key_alias.
        substring_matching: "true",
      },
    });
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export interface PaginatedKeyAliasResponse {
  aliases: string[];
  total_count: number;
  current_page: number;
  total_pages: number;
  size: number;
}

export const keyAliasesCall = async (
  accessToken: string,
  page: number = 1,
  size: number = 50,
  search?: string,
  team_id?: string,
): Promise<PaginatedKeyAliasResponse> => {
  /**
   * Get key aliases from proxy with pagination and optional search
   */
  try {
    return await apiClient.get(`/key/aliases`, {
      accessToken,
      query: {
        page: String(page),
        size: String(size),
        search: search || undefined,
        team_id: team_id || undefined,
      },
    });
  } catch (error) {
    console.error("Failed to fetch key aliases:", error);
    throw error;
  }
};

export const userDailyActivityAggregatedCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  userId: string | null = null,
) => {
  /**
   * Get aggregated daily user activity (no pagination)
   */
  try {
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };
    return await apiClient.get(`/user/daily/activity/aggregated`, {
      accessToken,
      query: {
        start_date: formatDate(startTime),
        end_date: formatDate(endTime),
        timezone: new Date().getTimezoneOffset().toString(),
        user_id: userId || undefined,
      },
    });
  } catch (error) {
    console.error("Failed to fetch aggregated user daily activity:", error);
    throw error;
  }
};

export const getPossibleUserRoles = async (accessToken: string) => {
  try {
    const data = (await apiClient.get(`/user/available_roles`, { accessToken })) as Record<
      string,
      Record<string, string>
    >;
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    throw error;
  }
};

export const teamCreateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    if (formValues.metadata) {
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const data = await apiClient.post(`/team/new`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const credentialCreateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    if (formValues.metadata) {
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const data = await apiClient.post(`/credentials`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const credentialListCall = async (accessToken: string) => {
  /**
   * Get all available teams on proxy
   */
  try {
    const data = await apiClient.get(`/credentials`, { accessToken });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const credentialGetCall = async (accessToken: string, credentialName: string | null, modelId: string | null) => {
  try {
    let path = `/credentials`;

    if (credentialName) {
      path += `/by_name/${credentialName}`;
    } else if (modelId) {
      path += `/by_model/${modelId}`;
    }

    const data = await apiClient.get(path, { accessToken });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const credentialDeleteCall = async (accessToken: string, credentialName: string) => {
  try {
    const data = await apiClient.delete(`/credentials/${credentialName}`, { accessToken });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to delete key:", error);
    throw error;
  }
};

export const credentialUpdateCall = async (
  accessToken: string,
  credentialName: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    if (formValues.metadata) {
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const data = await apiClient.patch(`/credentials/${credentialName}`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keyUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    if (formValues.model_tpm_limit) {
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.model_tpm_limit = JSON.parse(formValues.model_tpm_limit);
      } catch (error) {
        throw new Error("Failed to parse model_tpm_limit: " + error);
      }
    }

    if (formValues.model_rpm_limit) {
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.model_rpm_limit = JSON.parse(formValues.model_rpm_limit);
      } catch (error) {
        throw new Error("Failed to parse model_rpm_limit: " + error);
      }
    }
    const url = proxyBaseUrl ? `${proxyBaseUrl}/key/update` : `/key/update`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error(errorData);
    }
    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const teamUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/team/update` : `/team/update`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      NotificationsManager.fromBackend("Failed to update team settings: " + errorData);
      throw new Error(errorData);
    }
    const data = (await response.json()) as { data: Team; team_id: string };
    return data;
    // Handle success - you might want to update some state or UI based on the updated team
  } catch (error) {
    console.error("Failed to update team:", error);
    throw error;
  }
};

/**
 * Patch update a model
 *
 * @param accessToken
 * @param formValues
 * @returns
 */
export const modelPatchUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
  modelId: string,
) => {
  try {
    // Intentionally not logging the payload: it can contain freshly-entered
    // provider secrets (api_key, vertex_credentials, AWS creds).
    const url = proxyBaseUrl ? `${proxyBaseUrl}/model/${modelId}/update` : `/model/${modelId}/update`;
    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error update from the server:", errorData);
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to update model:", error);
    throw error;
  }
};

export interface Member {
  role: string;
  user_id: string | null;
  user_email?: string | null;
  max_budget_in_team?: number | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  budget_duration?: string | null;
  allowed_models?: string[] | null;
}

export const teamMemberAddCall = async (accessToken: string, teamId: string, formValues: Member) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/team/member_add` : `/team/member_add`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        team_id: teamId,
        member: formValues,
      }),
    });

    if (!response.ok) {
      // Read and parse JSON error body
      const errorText = await response.text();
      let parsedError: any = {};

      try {
        parsedError = JSON.parse(errorText);
      } catch (e) {
        console.warn("Failed to parse error body as JSON:", errorText);
      }

      const rawMessage = parsedError?.detail?.error || "Failed to add team member";
      const err = new Error(rawMessage);
      (err as any).raw = parsedError;
      throw err;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const teamBulkMemberAddCall = async (
  accessToken: string,
  teamId: string,
  members: Member[] | null,
  maxBudgetInTeam?: number,
  allUsers?: boolean,
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/team/bulk_member_add` : `/team/bulk_member_add`;

    let requestBody: any = {
      team_id: teamId,
    };

    if (allUsers) {
      requestBody.all_users = true;
    } else {
      requestBody.members = members;
    }

    if (maxBudgetInTeam !== undefined && maxBudgetInTeam !== null) {
      requestBody.max_budget_in_team = maxBudgetInTeam;
    }

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      // Read and parse JSON error body
      const errorText = await response.text();
      let parsedError: any = {};

      try {
        parsedError = JSON.parse(errorText);
      } catch (e) {
        console.warn("Failed to parse error body as JSON:", errorText);
      }

      const rawMessage = parsedError?.detail?.error || "Failed to bulk add team members";
      const err = new Error(rawMessage);
      (err as any).raw = parsedError;
      throw err;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to bulk add team members:", error);
    throw error;
  }
};

export const teamMemberUpdateCall = async (
  accessToken: string,
  teamId: string,
  formValues: Member, // Assuming formValues is an object
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/team/member_update` : `/team/member_update`;

    const requestBody: any = {
      team_id: teamId,
      role: formValues.role,
      user_id: formValues.user_id,
    };

    const orNull = (value: unknown) => (value === undefined || value === null || value === "" ? null : value);
    if (formValues.user_email !== undefined) {
      requestBody.user_email = formValues.user_email;
    }
    if ("max_budget_in_team" in formValues) {
      requestBody.max_budget_in_team = orNull(formValues.max_budget_in_team);
    }
    if ("tpm_limit" in formValues) {
      requestBody.tpm_limit = orNull(formValues.tpm_limit);
    }
    if ("rpm_limit" in formValues) {
      requestBody.rpm_limit = orNull(formValues.rpm_limit);
    }
    if ("budget_duration" in formValues) {
      requestBody.budget_duration = orNull(formValues.budget_duration);
    }
    if (formValues.allowed_models !== undefined) {
      requestBody.allowed_models = formValues.allowed_models;
    }

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      // Read and parse JSON error body
      const errorText = await response.text();
      let parsedError: any = {};

      try {
        parsedError = JSON.parse(errorText);
      } catch (e) {
        console.warn("Failed to parse error body as JSON:", errorText);
      }

      const rawMessage = parsedError?.detail?.error || "Failed to add team member";
      const err = new Error(rawMessage);
      (err as any).raw = parsedError;
      throw err;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to update team member:", error);
    throw error;
  }
};

export const teamMemberDeleteCall = async (
  accessToken: string,
  teamId: string,
  formValues: Member, // Assuming formValues is an object
) => {
  try {
    const data = await apiClient.post(`/team/member_delete`, {
      accessToken,
      body: {
        team_id: teamId,
        ...(formValues.user_email !== undefined && {
          user_email: formValues.user_email,
        }),
        ...(formValues.user_id !== undefined && {
          user_id: formValues.user_id,
        }),
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const organizationMemberAddCall = async (
  accessToken: string,
  organizationId: string,
  formValues: Member, // Assuming formValues is an object
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/organization/member_add` : `/organization/member_add`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        organization_id: organizationId,
        member: formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create organization member:", error);
    throw error;
  }
};

export const organizationMemberDeleteCall = async (accessToken: string, organizationId: string, userId: string) => {
  try {
    const data = await apiClient.delete(`/organization/member_delete`, {
      accessToken,
      body: {
        organization_id: organizationId,
        user_id: userId,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to delete organization member:", error);
    throw error;
  }
};
export const organizationMemberUpdateCall = async (
  accessToken: string,
  organizationId: string,
  formValues: Member, // Assuming formValues is an object
) => {
  try {
    const data = await apiClient.patch(`/organization/member_update`, {
      accessToken,
      body: {
        organization_id: organizationId,
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to update organization member:", error);
    throw error;
  }
};

export const userUpdateUserCall = async (
  accessToken: string,
  formValues: any, // Assuming formValues is an object
  userRole: string | null,
) => {
  try {
    const response_body = { ...formValues };
    if (userRole !== null) {
      response_body["user_role"] = userRole;
    }
    const data = (await apiClient.post(`/user/update`, { accessToken, body: response_body })) as {
      user_id: string;
      data: UserInfo;
    };
    //NotificationsManager.success("User role updated");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const userBulkUpdateUserCall = async (
  accessToken: string,
  formValues: any, // Assuming formValues is an object
  userIds?: string[], // Optional - if not provided, will update all users
  allUsers: boolean = false, // Flag to update all users
) => {
  try {
    let request_body: Record<string, any>;

    if (allUsers) {
      // Update all users mode
      request_body = {
        all_users: true,
        user_updates: formValues,
      };
    } else if (userIds && userIds.length > 0) {
      // Update specific users mode
      let users = [];
      for (const user_id of userIds) {
        users.push({
          user_id: user_id,
          ...formValues,
        });
      }
      request_body = {
        users: users,
      };
    } else {
      throw new Error("Must provide either userIds or set allUsers=true");
    }

    const data = (await apiClient.post(`/user/bulk_update`, { accessToken, body: request_body })) as {
      results: Array<{
        user_id?: string;
        user_email?: string;
        success: boolean;
        error?: string;
        updated_user?: any;
      }>;
      total_requested: number;
      successful_updates: number;
      failed_updates: number;
    };
    //NotificationsManager.success("User role updated");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const serviceHealthCheck = async (accessToken: string, service: string) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/health/services?service=${service}`
      : `/health/services?service=${service}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      // throw error with message
      throw new Error(errorData);
    }

    const data = await response.json();
    // You can add additional logic here based on the response if needed
    return data;
  } catch (error) {
    console.error("Failed to perform health check:", error);
    throw error;
  }
};

export const getBudgetList = async (accessToken: string) => {
  /**
   * Get all configurable params for setting a budget
   */
  try {
    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.get(`/budget/list`, { accessToken });
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};
export const getCallbacksCall = async (accessToken: string, userID: string, userRole: string) => {
  /**
   * Get all the models user has access to
   */
  try {
    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.get(`/get/config/callbacks`, { accessToken });
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const getGeneralSettingsCall = async (accessToken: string) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/list?config_type=general_settings`
      : `/config/list?config_type=general_settings`;

    //NotificationsManager.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const getRouterSettingsCall = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/router/settings`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to get router settings:", error);
    throw error;
  }
};

export const getCacheSettingsCall = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/cache/settings`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to get cache settings:", error);
    throw error;
  }
};

export const testCacheConnectionCall = async (accessToken: string, cacheSettings: Record<string, any>) => {
  try {
    const data = await apiClient.post(`/cache/settings/test`, {
      accessToken,
      body: {
        cache_settings: cacheSettings,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to test cache connection:", error);
    throw error;
  }
};

export const updateCacheSettingsCall = async (accessToken: string, cacheSettings: Record<string, any>) => {
  try {
    const data = await apiClient.post(`/cache/settings`, {
      accessToken,
      body: {
        cache_settings: cacheSettings,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to update cache settings:", error);
    throw error;
  }
};

export const getCoordinationRedisSettingsCall = async (
  accessToken: string,
): Promise<CoordinationRedisSettingsResponse> => {
  try {
    return await apiClient.get<CoordinationRedisSettingsResponse>(`/coordination_redis/settings`, { accessToken });
  } catch (error) {
    console.error("Failed to get coordination redis settings:", error);
    throw error;
  }
};

export const testCoordinationRedisConnectionCall = async (
  accessToken: string,
  settings: CoordinationRedisSettings,
): Promise<CoordinationRedisTestResponse> => {
  try {
    return await apiClient.post<CoordinationRedisTestResponse>(`/coordination_redis/settings/test`, {
      accessToken,
      body: { settings },
    });
  } catch (error) {
    console.error("Failed to test coordination redis connection:", error);
    throw error;
  }
};

export const updateCoordinationRedisSettingsCall = async (
  accessToken: string,
  settings: CoordinationRedisSettings,
): Promise<void> => {
  try {
    await apiClient.post(`/coordination_redis/settings`, {
      accessToken,
      body: { settings },
    });
  } catch (error) {
    console.error("Failed to update coordination redis settings:", error);
    throw error;
  }
};

export const getPassThroughEndpointsCall = async (accessToken: string, teamId?: string | null) => {
  try {
    let path = `/config/pass_through_endpoint`;

    if (teamId) {
      path += `/team/${teamId}`;
    }

    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.get(path, { accessToken });
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const getConfigFieldSetting = async (accessToken: string, fieldName: string) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/field/info?field_name=${fieldName}`
      : `/config/field/info?field_name=${fieldName}`;

    //NotificationsManager.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to set callbacks:", error);
    throw error;
  }
};

export const createPassThroughEndpoint = async (accessToken: string, formValues: Record<string, any>) => {
  /**
   * Set callbacks on proxy
   */
  try {
    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.post(`/config/pass_through_endpoint`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to set callbacks:", error);
    throw error;
  }
};

export const updateConfigFieldSetting = async (accessToken: string, fieldName: string, fieldValue: any) => {
  try {
    let formData = {
      field_name: fieldName,
      field_value: fieldValue,
      config_type: "general_settings",
    };
    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.post(`/config/field/update`, { accessToken, body: formData });
    //NotificationsManager.info("Received model data");
    NotificationsManager.success("Successfully updated value!");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to set callbacks:", error);
    throw error;
  }
};

export const deleteConfigFieldSetting = async (accessToken: string, fieldName: string) => {
  try {
    let formData = {
      field_name: fieldName,
      config_type: "general_settings",
    };
    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.post(`/config/field/delete`, { accessToken, body: formData });
    NotificationsManager.success("Field reset on proxy");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const deletePassThroughEndpointsCall = async (accessToken: string, endpointId: string) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/pass_through_endpoint?endpoint_id=${endpointId}`
      : `/config/pass_through_endpoint?endpoint_id=${endpointId}`;

    //NotificationsManager.info("Requesting model data");
    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const setCallbacksCall = async (accessToken: string, formValues: Record<string, any>) => {
  /**
   * Set callbacks on proxy
   */
  try {
    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.post(`/config/update`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to set callbacks:", error);
    throw error;
  }
};

export const individualModelHealthCheckCall = async (accessToken: string, modelId: string) => {
  /**
   * Run health check for a specific model using model ID (so each deployment is checked separately).
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/health?model_id=${encodeURIComponent(modelId)}`
      : `/health?model_id=${encodeURIComponent(modelId)}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to call /health for model id ${modelId}:`, error);
    throw error;
  }
};

export const cachingHealthCheckCall = async (accessToken: string) => {
  /**
   * Get all the models user has access to
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/cache/ping` : `/cache/ping`;

    //NotificationsManager.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to call /cache/ping:", error);
    throw error;
  }
};

export const latestHealthChecksCall = async (accessToken: string) => {
  /**
   * Get the latest health check status for all models
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/health/latest` : `/health/latest`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to call /health/latest:", error);
    throw error;
  }
};

export const getProxyUISettings = async (accessToken: string) => {
  /**
   * Get all the models user has access to
   */
  try {
    //NotificationsManager.info("Requesting model data");
    const data = await apiClient.get(`/sso/get/ui_settings`, { accessToken });
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const getUISettings = async (accessToken: string) => {
  /**
   * Get UI-specific configuration flags from the database
   */
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/get/ui_settings` : `/get/ui_settings`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      console.error("Failed to get UI settings:", errorMessage);
      return null;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get UI settings:", error);
    return null;
  }
};

export const getMCPSemanticFilterSettings = async (accessToken: string) => {
  /**
   * Get MCP semantic filter configuration
   */
  try {
    const data = await apiClient.get(`/get/mcp_semantic_filter_settings`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to get MCP semantic filter settings:", error);
    throw error;
  }
};

export const updateMCPSemanticFilterSettings = async (accessToken: string, settings: Record<string, any>) => {
  /**
   * Update MCP semantic filter settings
   * Settings will be applied across all pods within 10 seconds
   */
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/update/mcp_semantic_filter_settings`
      : `/update/mcp_semantic_filter_settings`;
    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(settings),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to update MCP semantic filter settings:", error);
    throw error;
  }
};

export const testMCPSemanticFilter = async (accessToken: string, model: string, query: string) => {
  /**
   * Test MCP semantic filter by making a responses API call
   * Returns both the response data and headers containing filter information
   */
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/responses` : `/v1/responses`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: model,
        input: [
          {
            role: "user",
            content: query,
            type: "message",
          },
        ],
        tools: [
          {
            type: "mcp",
            server_url: "litellm_proxy",
            require_approval: "never",
          },
        ],
        tool_choice: "required",
      }),
    });

    // Extract headers before checking response status
    const filterHeader = response.headers.get("x-litellm-semantic-filter");
    const toolsHeader = response.headers.get("x-litellm-semantic-filter-tools");

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();

    // Return both data and headers
    return {
      data,
      headers: {
        filter: filterHeader,
        tools: toolsHeader,
      },
    };
  } catch (error) {
    console.error("Failed to test MCP semantic filter:", error);
    throw error;
  }
};

export const getGuardrailsList = async (accessToken: string) => {
  try {
    const v2Url = proxyBaseUrl ? `${proxyBaseUrl}/v2/guardrails/list` : `/v2/guardrails/list`;
    const response = await fetch(v2Url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`v2 guardrails/list returned ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    try {
      const v1Url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/list` : `/guardrails/list`;
      const fallbackResponse = await fetch(v1Url, {
        method: "GET",
        headers: {
          [globalLitellmHeaderName]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });

      if (!fallbackResponse.ok) {
        const errorData = await fallbackResponse.json();
        const errorMessage = deriveErrorMessage(errorData);
        handleError(errorMessage);
        throw new Error(errorMessage);
      }

      return await fallbackResponse.json();
    } catch (fallbackError) {
      console.error("Failed to get guardrails list:", fallbackError);
      throw fallbackError;
    }
  }
};

// Team guardrail submissions (admin)
export interface GuardrailSubmissionItem {
  guardrail_id: string;
  guardrail_name: string;
  status: string; // "pending_review" | "active" | "rejected"
  team_id?: string | null;
  team_guardrail?: boolean; // true when submitted via team (team_id set)
  litellm_params?: Record<string, unknown> | null;
  guardrail_info?: Record<string, unknown> | null;
  submitted_by_user_id?: string | null;
  submitted_by_email?: string | null;
  submitted_at?: string | null;
  reviewed_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

interface GuardrailSubmissionSummary {
  total: number;
  pending_review: number;
  active: number;
  rejected: number;
}

interface ListGuardrailSubmissionsResponse {
  submissions: GuardrailSubmissionItem[];
  summary: GuardrailSubmissionSummary;
}

export const listGuardrailSubmissions = async (
  accessToken: string,
  params?: { status?: string; team_id?: string; team_guardrail?: boolean; search?: string },
): Promise<ListGuardrailSubmissionsResponse> => {
  return apiClient.get<ListGuardrailSubmissionsResponse>(`/guardrails/submissions`, {
    accessToken,
    query: {
      ...(params?.status ? { status: params.status } : {}),
      ...(params?.team_id ? { team_id: params.team_id } : {}),
      ...(params?.team_guardrail !== undefined ? { team_guardrail: params.team_guardrail } : {}),
      ...(params?.search ? { search: params.search } : {}),
    },
  });
};

export const approveGuardrailSubmission = async (
  accessToken: string,
  guardrailId: string,
): Promise<{ guardrail_id: string; status: string; message: string }> => {
  return apiClient.post<{ guardrail_id: string; status: string; message: string }>(
    `/guardrails/submissions/${encodeURIComponent(guardrailId)}/approve`,
    { accessToken },
  );
};

export const rejectGuardrailSubmission = async (
  accessToken: string,
  guardrailId: string,
): Promise<{ guardrail_id: string; status: string; message: string }> => {
  return apiClient.post<{ guardrail_id: string; status: string; message: string }>(
    `/guardrails/submissions/${encodeURIComponent(guardrailId)}/reject`,
    { accessToken },
  );
};

// Guardrails / Policies usage (dashboard)
export const getGuardrailsUsageOverview = async (accessToken: string, startDate?: string, endDate?: string) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/usage/overview` : `/guardrails/usage/overview`;
    const params = new URLSearchParams();
    if (startDate) params.append("start_date", startDate);
    if (endDate) params.append("end_date", endDate);
    if (params.toString()) url += `?${params.toString()}`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(deriveErrorMessage(errorData));
    }
    return response.json();
  } catch (error) {
    console.error("Failed to get guardrails usage overview:", error);
    throw error;
  }
};

export const getGuardrailsUsageDetail = async (
  accessToken: string,
  guardrailId: string,
  startDate?: string,
  endDate?: string,
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/guardrails/usage/detail/${encodeURIComponent(guardrailId)}`
      : `/guardrails/usage/detail/${encodeURIComponent(guardrailId)}`;
    const params = new URLSearchParams();
    if (startDate) params.append("start_date", startDate);
    if (endDate) params.append("end_date", endDate);
    if (params.toString()) url += `?${params.toString()}`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(deriveErrorMessage(errorData));
    }
    return response.json();
  } catch (error) {
    console.error("Failed to get guardrails usage detail:", error);
    throw error;
  }
};

export const getGuardrailsUsageLogs = async (
  accessToken: string,
  options: {
    guardrailId?: string;
    policyId?: string;
    page?: number;
    pageSize?: number;
    action?: string;
    startDate?: string;
    endDate?: string;
  },
) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/usage/logs` : `/guardrails/usage/logs`;
    const params = new URLSearchParams();
    if (options.guardrailId) params.append("guardrail_id", options.guardrailId);
    if (options.policyId) params.append("policy_id", options.policyId);
    if (options.page != null) params.append("page", String(options.page));
    if (options.pageSize != null) params.append("page_size", String(options.pageSize));
    if (options.action) params.append("action", options.action);
    if (options.startDate) params.append("start_date", options.startDate);
    if (options.endDate) params.append("end_date", options.endDate);
    if (params.toString()) url += `?${params.toString()}`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(deriveErrorMessage(errorData));
    }
    return response.json();
  } catch (error) {
    console.error("Failed to get guardrails usage logs:", error);
    throw error;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Policy CRUD API Calls
// ─────────────────────────────────────────────────────────────────────────────

export const getPoliciesList = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/policies/list`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to get policies list:", error);
    throw error;
  }
};

interface GuardrailInputs {
  texts?: string[];
  images?: string[];
  [key: string]: unknown;
}

interface TestPoliciesAndGuardrailsRequest {
  policy_names?: string[] | null;
  guardrail_names?: string[] | null;
  /** Single input (legacy). Use inputs_list for per-input batch processing. */
  inputs?: GuardrailInputs | null;
  /** List of inputs; each processed separately for batch compliance testing. */
  inputs_list?: GuardrailInputs[] | null;
  request_data?: Record<string, unknown>;
  input_type?: "request" | "response";
  /** When set, backend runs chat completion with this model/agent per input and includes agent_response in each result. */
  agent_id?: string | null;
}

interface GuardrailErrorEntry {
  guardrail_name: string;
  message: string;
}

interface TestPoliciesAndGuardrailsResultItem {
  inputs: Record<string, unknown>;
  guardrail_errors: GuardrailErrorEntry[];
  /** Present when request included agent_id; serialized chat completion response. */
  agent_response?: Record<string, unknown>;
}

interface TestPoliciesAndGuardrailsResponse {
  inputs?: Record<string, unknown>;
  guardrail_errors?: GuardrailErrorEntry[];
  /** Present when inputs_list was used; one result per input. */
  results?: TestPoliciesAndGuardrailsResultItem[];
}

export const testPoliciesAndGuardrails = async (
  accessToken: string,
  body: TestPoliciesAndGuardrailsRequest,
  signal?: AbortSignal,
): Promise<TestPoliciesAndGuardrailsResponse> => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/utils/test_policies_and_guardrails`
      : `/utils/test_policies_and_guardrails`;
    const response = await fetch(url, {
      method: "POST",
      signal,
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        policy_names: body.policy_names ?? null,
        guardrail_names: body.guardrail_names ?? null,
        inputs: body.inputs ?? null,
        inputs_list: body.inputs_list ?? null,
        request_data: body.request_data ?? {},
        input_type: body.input_type ?? "request",
        agent_id: body.agent_id ?? null,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      let errorMessage = "Failed to test policies and guardrails";
      try {
        const errorJson = JSON.parse(errorData);
        if (errorJson.detail)
          errorMessage = typeof errorJson.detail === "string" ? errorJson.detail : JSON.stringify(errorJson.detail);
        else if (errorJson.message) errorMessage = errorJson.message;
      } catch {
        errorMessage = errorData || errorMessage;
      }
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to test policies and guardrails:", error);
    throw error;
  }
};

export const getPolicyInfoWithGuardrails = async (accessToken: string, policyName: string) => {
  try {
    const data = await apiClient.get(`/policy/info/${policyName}`, { accessToken });
    return data;
  } catch (error) {
    console.error(`Failed to get policy info for ${policyName}:`, error);
    throw error;
  }
};

export const getPolicyTemplates = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/policy/templates`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to get policy templates:", error);
    throw error;
  }
};

export const enrichPolicyTemplate = async (
  accessToken: string,
  templateId: string,
  parameters: Record<string, string>,
  model?: string,
  competitors?: string[],
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/policy/templates/enrich` : `/policy/templates/enrich`;
    const body: any = { template_id: templateId, parameters };
    if (model) body.model = model;
    if (competitors) body.competitors = competitors;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to enrich policy template:", error);
    throw error;
  }
};

export const suggestPolicyTemplates = async (
  accessToken: string,
  attackExamples: string[],
  description: string,
  model: string,
) => {
  try {
    return await apiClient.post(`/policy/templates/suggest`, {
      accessToken,
      body: {
        attack_examples: attackExamples.filter((e) => e.trim()),
        description,
        model,
      },
    });
  } catch (error) {
    console.error("Failed to suggest policy templates:", error);
    throw error;
  }
};

export const testPolicyTemplate = async (accessToken: string, guardrailDefinitions: any[], text: string) => {
  try {
    return await apiClient.post(`/policy/templates/test`, {
      accessToken,
      body: {
        guardrail_definitions: guardrailDefinitions,
        text,
      },
    });
  } catch (error) {
    console.error("Failed to test policy template:", error);
    throw error;
  }
};

export const enrichPolicyTemplateStream = async (
  accessToken: string,
  templateId: string,
  parameters: Record<string, string>,
  model: string,
  onCompetitor: (name: string) => void,
  onDone: (result: {
    competitors: string[];
    competitor_variations: Record<string, string[]>;
    guardrailDefinitions: any[];
  }) => void,
  onError?: (error: string) => void,
  options?: { instruction?: string; existingCompetitors?: string[] },
  onStatus?: (message: string) => void,
) => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/policy/templates/enrich/stream` : `/policy/templates/enrich/stream`;
  const body: any = { template_id: templateId, parameters, model };
  if (options?.instruction) body.instruction = options.instruction;
  if (options?.existingCompetitors) body.competitors = options.existingCompetitors;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    handleError(errorMessage);
    throw new Error(errorMessage);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  // eslint-disable-next-line no-constant-condition -- stream read loop
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const event = JSON.parse(line.slice(6));
        if (event.type === "competitor") {
          onCompetitor(event.name);
        } else if (event.type === "status") {
          onStatus?.(event.message);
        } else if (event.type === "done") {
          onDone(event);
        } else if (event.type === "error") {
          onError?.(event.message);
        }
      } catch {
        // skip malformed events
      }
    }
  }
};

export interface UsageAiToolCallEvent {
  tool_name: string;
  tool_label: string;
  arguments: Record<string, string>;
  status: "running" | "complete" | "error";
  error?: string;
}

export const usageAiChatStream = async (
  accessToken: string,
  messages: { role: string; content: string }[],
  model: string,
  onChunk: (content: string) => void,
  onDone: () => void,
  onError?: (error: string) => void,
  onStatus?: (message: string) => void,
  onToolCall?: (event: UsageAiToolCallEvent) => void,
  signal?: AbortSignal,
) => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/usage/ai/chat` : `/usage/ai/chat`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ messages, model }),
    signal,
  });

  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    handleError(errorMessage);
    throw new Error(errorMessage);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  // eslint-disable-next-line no-constant-condition -- stream read loop
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const event = JSON.parse(line.slice(6));
        if (event.type === "chunk") {
          onChunk(event.content);
        } else if (event.type === "status") {
          onStatus?.(event.message);
        } else if (event.type === "tool_call") {
          onToolCall?.(event as UsageAiToolCallEvent);
        } else if (event.type === "done") {
          onDone();
        } else if (event.type === "error") {
          onError?.(event.message);
        }
      } catch {
        // skip malformed events
      }
    }
  }
};

export const createPolicyCall = async (accessToken: string, policyData: any) => {
  try {
    const data = await apiClient.post(`/policies`, { accessToken, body: policyData });
    return data;
  } catch (error) {
    console.error("Failed to create policy:", error);
    throw error;
  }
};

export const updatePolicyCall = async (accessToken: string, policyId: string, policyData: any) => {
  try {
    const data = await apiClient.put(`/policies/${policyId}`, { accessToken, body: policyData });
    return data;
  } catch (error) {
    console.error("Failed to update policy:", error);
    throw error;
  }
};

export const listPolicyVersions = async (
  accessToken: string,
  policyName: string,
): Promise<{ policy_name: string; versions: any[]; total_count: number }> => {
  try {
    const encodedName = encodeURIComponent(policyName);
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/policies/name/${encodedName}/versions`
      : `/policies/name/${encodedName}/versions`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to list policy versions:", error);
    throw error;
  }
};

export const createPolicyVersion = async (
  accessToken: string,
  policyName: string,
  sourcePolicyId?: string | null,
): Promise<any> => {
  try {
    const encodedName = encodeURIComponent(policyName);
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/policies/name/${encodedName}/versions`
      : `/policies/name/${encodedName}/versions`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ source_policy_id: sourcePolicyId ?? undefined }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to create policy version:", error);
    throw error;
  }
};

export const updatePolicyVersionStatus = async (
  accessToken: string,
  policyId: string,
  versionStatus: "published" | "production",
): Promise<any> => {
  try {
    return await apiClient.put(`/policies/${policyId}/status`, {
      accessToken,
      body: { version_status: versionStatus },
    });
  } catch (error) {
    console.error("Failed to update policy version status:", error);
    throw error;
  }
};

export const deletePolicyCall = async (accessToken: string, policyId: string) => {
  try {
    const data = await apiClient.delete(`/policies/${policyId}`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to delete policy:", error);
    throw error;
  }
};

export const getPolicyInfo = async (accessToken: string, policyId: string) => {
  try {
    const data = await apiClient.get(`/policies/${policyId}`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to get policy info:", error);
    throw error;
  }
};

// Policy Attachments API Calls

export const getPolicyAttachmentsList = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/policies/attachments/list`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to get policy attachments list:", error);
    throw error;
  }
};

export const createPolicyAttachmentCall = async (accessToken: string, attachmentData: any) => {
  try {
    const data = await apiClient.post(`/policies/attachments`, { accessToken, body: attachmentData });
    return data;
  } catch (error) {
    console.error("Failed to create policy attachment:", error);
    throw error;
  }
};

export const deletePolicyAttachmentCall = async (accessToken: string, attachmentId: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/policies/attachments/${attachmentId}`
      : `/policies/attachments/${attachmentId}`;
    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to delete policy attachment:", error);
    throw error;
  }
};

export const testPipelineCall = async (
  accessToken: string,
  pipeline: any,
  testMessages: Array<{ role: string; content: string }>,
) => {
  try {
    const data = await apiClient.post(`/policies/test-pipeline`, {
      accessToken,
      body: { pipeline, test_messages: testMessages },
    });
    return data;
  } catch (error) {
    console.error("Failed to test pipeline:", error);
    throw error;
  }
};

export const getResolvedGuardrails = async (accessToken: string, policyId: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/policies/${policyId}/resolved-guardrails`
      : `/policies/${policyId}/resolved-guardrails`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get resolved guardrails:", error);
    throw error;
  }
};

export const resolvePoliciesCall = async (
  accessToken: string,
  context: { team_alias?: string; key_alias?: string; model?: string; tags?: string[] },
) => {
  try {
    return await apiClient.post(`/policies/resolve`, { accessToken, body: context });
  } catch (error) {
    console.error("Failed to resolve policies:", error);
    throw error;
  }
};

export const estimateAttachmentImpactCall = async (accessToken: string, attachmentData: any) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/policies/attachments/estimate-impact`
      : `/policies/attachments/estimate-impact`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(attachmentData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to estimate attachment impact:", error);
    throw error;
  }
};

export const getPromptsList = async (accessToken: string, environment?: string): Promise<ListPromptsResponse> => {
  try {
    return await apiClient.get(`/prompts/list`, { accessToken, query: { environment: environment || undefined } });
  } catch (error) {
    console.error("Failed to get prompts list:", error);
    throw error;
  }
};

export const getPromptInfo = async (
  accessToken: string,
  promptId: string,
  environment?: string,
): Promise<PromptInfoResponse> => {
  try {
    return await apiClient.get(`/prompts/${promptId}/info`, {
      accessToken,
      query: { environment: environment || undefined },
    });
  } catch (error) {
    console.error("Failed to get prompt info:", error);
    throw error;
  }
};

export const getPromptVersions = async (
  accessToken: string,
  promptId: string,
  environment?: string,
): Promise<ListPromptsResponse> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/prompts/${promptId}/versions` : `/prompts/${promptId}/versions`;
    if (environment) {
      url += `?environment=${encodeURIComponent(environment)}`;
    }
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      // Don't throw global error for 404 (no versions found) as we might want to handle it gracefully
      if (response.status !== 404) {
        handleError(errorMessage);
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get prompt versions:", error);
    throw error;
  }
};

export const createPromptCall = async (accessToken: string, promptData: any) => {
  try {
    const data = await apiClient.post(`/prompts`, { accessToken, body: promptData });
    return data;
  } catch (error) {
    console.error("Failed to create prompt:", error);
    throw error;
  }
};

export const updatePromptCall = async (accessToken: string, promptId: string, promptData: any) => {
  try {
    const data = await apiClient.put(`/prompts/${promptId}`, { accessToken, body: promptData });
    return data;
  } catch (error) {
    console.error("Failed to update prompt:", error);
    throw error;
  }
};

export const deletePromptCall = async (accessToken: string, promptId: string) => {
  try {
    const data = await apiClient.delete(`/prompts/${promptId}`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to delete prompt:", error);
    throw error;
  }
};

export const convertPromptFileToJson = async (
  accessToken: string,
  file: File,
): Promise<{ prompt_id: string; json_data: any }> => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const url = proxyBaseUrl ? `${proxyBaseUrl}/utils/dotprompt_json_converter` : `/utils/dotprompt_json_converter`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to convert prompt file:", error);
    throw error;
  }
};

export const createAgentCall = async (accessToken: string, agentData: any) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/agents` : `/v1/agents`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...agentData,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to create agent:", error);
    throw error;
  }
};

export interface DiscoveredAgentCard {
  protocolVersion?: string;
  name?: string;
  description?: string;
  version?: string;
  url?: string;
  iconUrl?: string;
  documentationUrl?: string;
  defaultInputModes?: string[];
  defaultOutputModes?: string[];
  capabilities?: Record<string, any>;
  skills?: Array<{
    id?: string;
    name?: string;
    description?: string;
    tags?: string[];
    examples?: string[];
    [key: string]: any;
  }>;
  provider?: { organization?: string; url?: string };
  [key: string]: any;
}

export interface DiscoverAgentCardResponse {
  url: string;
  agent_card: DiscoveredAgentCard;
}

/**
 * How the backend should locate the upstream agent card.
 *
 * - ``well_known_fallback`` (default): pure A2A — try the three standard
 *   well-known paths under the base URL.
 * - ``langgraph_platform``: LangGraph Platform — hits the canonical
 *   well-known path with an ``assistant_id`` query parameter, because
 *   LangGraph mounts one shared card endpoint per deployment.
 */
export type DiscoveryMode = "well_known_fallback" | "langgraph_platform";

export interface DiscoverAgentCardOptions {
  discovery_mode?: DiscoveryMode;
  /** Mode-specific params. ``langgraph_platform`` requires ``assistant_id``. */
  params?: Record<string, any>;
}

export const discoverAgentCardCall = async (
  accessToken: string,
  url: string,
  options?: DiscoverAgentCardOptions,
): Promise<DiscoverAgentCardResponse> => {
  const endpoint = proxyBaseUrl ? `${proxyBaseUrl}/v1/a2a/discover` : `/v1/a2a/discover`;
  const body: Record<string, any> = { url };
  if (options?.discovery_mode) body.discovery_mode = options.discovery_mode;
  if (options?.params) body.params = options.params;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error(errorData);
  }

  return (await response.json()) as DiscoverAgentCardResponse;
};

export const createGuardrailCall = async (accessToken: string, guardrailData: any) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails` : `/guardrails`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        guardrail: guardrailData,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to create guardrail:", error);
    throw error;
  }
};

export const uiSpendLogDetailsCall = async (accessToken: string, logId: string, start_date: string) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/spend/logs/ui/${logId}?start_date=${encodeURIComponent(start_date)}`
      : `/spend/logs/ui/${logId}?start_date=${encodeURIComponent(start_date)}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch log details:", error);
    throw error;
  }
};

export const getInternalUserSettings = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/get/internal_user_settings`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to fetch SSO settings:", error);
    throw error;
  }
};

export const updateInternalUserSettings = async (accessToken: string, settings: Record<string, any>) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl ? `${proxyBaseUrl}/update/internal_user_settings` : `/update/internal_user_settings`;

    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(settings),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    NotificationsManager.success("Internal user settings updated successfully");
    return data;
  } catch (error) {
    console.error("Failed to update internal user settings:", error);
    throw error;
  }
};

export const fetchOpenAPIRegistry = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/mcp/openapi-registry` : `/v1/mcp/openapi-registry`;

    const response = await fetch(url, {
      method: HTTP_REQUEST.GET,
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(deriveErrorMessage(errorData));
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to fetch OpenAPI registry:", error);
    throw error;
  }
};

export const fetchDiscoverableMCPServers = async (accessToken: string) => {
  try {
    return await apiClient.get(`/v1/mcp/discover`, { accessToken });
  } catch (error) {
    console.error("Failed to fetch discoverable MCP servers:", error);
    throw error;
  }
};

export const fetchMCPServers = async (accessToken: string, teamId?: string | null) => {
  try {
    return await apiClient.get(`/v1/mcp/server`, { accessToken, query: { team_id: teamId || undefined } });
  } catch (error) {
    console.error("Failed to fetch MCP servers:", error);
    throw error;
  }
};

export const fetchMCPServerHealth = async (accessToken: string, serverIds?: string[]) => {
  try {
    return await apiClient.get(`/v1/mcp/server/health`, {
      accessToken,
      query: {
        server_ids: serverIds && serverIds.length > 0 ? serverIds : undefined,
      },
    });
  } catch (error) {
    console.error("Failed to fetch MCP server health:", error);
    throw error;
  }
};

export const fetchMCPAccessGroups = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/v1/mcp/access_groups`, { accessToken });
    return data.access_groups || [];
  } catch (error) {
    console.error("Failed to fetch MCP access groups:", error);
    throw error;
  }
};

export const fetchMCPClientIp = async (accessToken: string): Promise<string | null> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/mcp/network/client-ip` : `/v1/mcp/network/client-ip`;

    const response = await fetch(url, {
      method: HTTP_REQUEST.GET,
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    return data.ip || null;
  } catch {
    return null;
  }
};

export const createMCPServer = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    const data = await apiClient.post(`/v1/mcp/server`, {
      accessToken,
      body: {
        ...formValues, // Include formValues in the request body
      },
    });
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const updateMCPServer = async (accessToken: string, formValues: Record<string, any>) => {
  try {
    return await apiClient.put(`/v1/mcp/server`, { accessToken, body: formValues });
  } catch (error) {
    console.error("Failed to update MCP server:", error);
    throw error;
  }
};

export const deleteMCPServer = async (accessToken: string, serverId: string) => {
  try {
    await apiClient.delete(`/v1/mcp/server/${serverId}`, { accessToken });
  } catch (error) {
    console.error("Failed to delete key:", error);
    throw error;
  }
};

export const fetchMCPToolsets = async (accessToken: string): Promise<any[]> => {
  try {
    return await apiClient.get(`/v1/mcp/toolset`, { accessToken });
  } catch (error) {
    console.error("Failed to fetch MCP toolsets:", error);
    throw error;
  }
};

export const createMCPToolset = async (accessToken: string, formValues: Record<string, any>) => {
  try {
    return await apiClient.post(`/v1/mcp/toolset`, { accessToken, body: formValues });
  } catch (error) {
    console.error("Failed to create MCP toolset:", error);
    throw error;
  }
};

export const updateMCPToolset = async (accessToken: string, formValues: Record<string, any>) => {
  try {
    return await apiClient.put(`/v1/mcp/toolset`, { accessToken, body: formValues });
  } catch (error) {
    console.error("Failed to update MCP toolset:", error);
    throw error;
  }
};

export const deleteMCPToolset = async (accessToken: string, toolsetId: string) => {
  try {
    await apiClient.delete(`/v1/mcp/toolset/${toolsetId}`, { accessToken });
  } catch (error) {
    console.error("Failed to delete MCP toolset:", error);
    throw error;
  }
};

export const registerMCPServer = async (accessToken: string, formValues: Record<string, any>) => {
  try {
    return await apiClient.post(`/v1/mcp/server/register`, { accessToken, body: formValues });
  } catch (error) {
    console.error("Failed to register MCP server:", error);
    throw error;
  }
};

export const fetchMCPSubmissions = async (accessToken: string) => {
  try {
    const url = (proxyBaseUrl ? `${proxyBaseUrl}` : "") + `/v1/mcp/server/submissions`;
    const response = await fetch(url, {
      method: HTTP_REQUEST.GET,
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }
    return response.json();
  } catch (error) {
    console.error("Failed to fetch MCP submissions:", error);
    throw error;
  }
};

export const approveMCPServer = async (accessToken: string, serverId: string) => {
  try {
    const url = (proxyBaseUrl ? `${proxyBaseUrl}` : "") + `/v1/mcp/server/${encodeURIComponent(serverId)}/approve`;
    const response = await fetch(url, {
      method: HTTP_REQUEST.PUT,
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }
    return response.json();
  } catch (error) {
    console.error("Failed to approve MCP server:", error);
    throw error;
  }
};

export const rejectMCPServer = async (accessToken: string, serverId: string, reviewNotes?: string) => {
  try {
    const url = (proxyBaseUrl ? `${proxyBaseUrl}` : "") + `/v1/mcp/server/${encodeURIComponent(serverId)}/reject`;
    const response = await fetch(url, {
      method: HTTP_REQUEST.PUT,
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ review_notes: reviewNotes ?? null }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }
    return response.json();
  } catch (error) {
    console.error("Failed to reject MCP server:", error);
    throw error;
  }
};

// Search Tools API calls
export const fetchSearchTools = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/search_tools/list`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to fetch search tools:", error);
    throw error;
  }
};

export const createSearchTool = async (accessToken: string, formValues: Record<string, any>) => {
  try {
    const data = await apiClient.post(`/search_tools`, {
      accessToken,
      body: {
        search_tool: formValues,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to create search tool:", error);
    throw error;
  }
};

export const updateSearchTool = async (accessToken: string, searchToolId: string, formValues: Record<string, any>) => {
  try {
    const data = await apiClient.put(`/search_tools/${searchToolId}`, {
      accessToken,
      body: {
        search_tool: formValues,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to update search tool:", error);
    throw error;
  }
};

export const deleteSearchTool = async (accessToken: string, searchToolId: string) => {
  try {
    const data = await apiClient.delete(`/search_tools/${searchToolId}`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to delete search tool:", error);
    throw error;
  }
};

export const fetchAvailableSearchProviders = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/search_tools/ui/available_providers`
      : `/search_tools/ui/available_providers`;

    const response = await fetch(url, {
      method: HTTP_REQUEST.GET,
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch available search providers:", error);
    throw error;
  }
};

export const testSearchToolConnection = async (accessToken: string, litellmParams: Record<string, any>) => {
  try {
    const data = await apiClient.post(`/search_tools/test_connection`, {
      accessToken,
      body: {
        litellm_params: litellmParams,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to test search tool connection:", error);
    throw error;
  }
};

export const listMCPTools = async (
  accessToken: string,
  serverId: string,
  customHeaders?: Record<string, string>,
  includeDisabledTools?: boolean,
) => {
  // Construct base URL. include_disabled_tools returns the full server catalog
  // (admin-only, backend-enforced) so the settings UI can configure the allowlist.
  const query = `server_id=${serverId}${includeDisabledTools ? "&include_disabled_tools=true" : ""}`;
  let url = proxyBaseUrl ? `${proxyBaseUrl}/mcp-rest/tools/list?${query}` : `/mcp-rest/tools/list?${query}`;

  const headers: Record<string, string> = {
    [globalLitellmHeaderName]: `Bearer ${accessToken}`,
    "Content-Type": "application/json",
    ...customHeaders, // Merge custom headers for passthrough auth
  };

  let response: Response;
  try {
    response = await fetch(url, {
      method: "GET",
      headers,
    });
  } catch (error) {
    // Network-level failure (no HTTP response). Preserve legacy shape so the
    // caller can render a generic error message without crashing.
    console.error("Failed to fetch MCP tools (network error):", error);
    return {
      tools: [],
      error: "network_error",
      message: error instanceof Error ? error.message : "Failed to fetch MCP tools",
      stack_trace: null,
    };
  }

  let data: any = null;
  try {
    data = await response.json();
  } catch (parseError) {
    console.error("Failed to parse MCP tools response:", parseError);
    return {
      tools: [],
      error: "parse_error",
      message: "Failed to parse MCP tools response",
      status: response.status,
      statusText: response.statusText,
      stack_trace: null,
    };
  }

  if (!response.ok) {
    // Preserve the legacy "never throws" contract so existing callers
    // (e.g. MCPToolPermissions, MCPAppsPanel, MCPConnectPicker) can continue
    // to inspect `result.error` / `result.message`. Attach `status` so
    // callers that need to react to auth failures (e.g. the useQuery in
    // mcp_tools.tsx) can still detect 401s from the returned object.
    const errorMessage = (data && (data.message || data.error)) || "Failed to fetch MCP tools";
    return {
      tools: [],
      error: (data && data.error) || `http_${response.status}`,
      message: errorMessage,
      status: response.status,
      statusText: response.statusText,
      details: data,
      stack_trace: null,
    };
  }

  // Return the full response object which includes tools, error, message, and stack_trace
  return data;
};

interface CallMCPToolOptions {
  guardrails?: string[];
  customHeaders?: Record<string, string>;
}

export const callMCPTool = async (
  accessToken: string,
  serverId: string,
  toolName: string,
  toolArguments: Record<string, any>,
  options?: CallMCPToolOptions,
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl ? `${proxyBaseUrl}/mcp-rest/tools/call` : `/mcp-rest/tools/call`;

    const headers: Record<string, string> = {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      ...(options?.customHeaders || {}), // Merge custom headers for passthrough auth
    };

    const body: Record<string, any> = {
      server_id: serverId,
      name: toolName,
      arguments: toolArguments,
    };
    if (options?.guardrails && options.guardrails.length > 0) {
      body.litellm_metadata = { guardrails: options.guardrails };
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      let errorMessage = "Network response was not ok";
      let errorDetails = null;

      // First, try to get the response as text to see what we're dealing with
      const responseText = await response.text();

      try {
        // Try to parse as JSON
        const errorData = JSON.parse(responseText);

        if (errorData.detail) {
          if (typeof errorData.detail === "string") {
            errorMessage = errorData.detail;
          } else if (typeof errorData.detail === "object") {
            errorMessage = errorData.detail.message || errorData.detail.error || "An error occurred";
            errorDetails = errorData.detail;
          }
        } else {
          errorMessage = errorData.message || errorData.error || errorMessage;
        }
      } catch (parseError) {
        console.error("Failed to parse JSON error response:", parseError);
        // If JSON parsing fails, use the raw text
        if (responseText) {
          errorMessage = responseText;
        }
      }

      // Create a more informative error object
      const enhancedError = new Error(errorMessage);
      (enhancedError as any).status = response.status;
      (enhancedError as any).statusText = response.statusText;
      (enhancedError as any).details = errorDetails;

      handleError(errorMessage);
      throw enhancedError;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to call MCP tool:", error);
    console.error("Error type:", typeof error);
    if (error instanceof Error) {
      console.error("Error message:", error.message);
      console.error("Error stack:", error.stack);
    }
    throw error;
  }
};

export const tagCreateCall = async (accessToken: string, formValues: TagNewRequest): Promise<void> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/new` : `/tag/new`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(formValues),
    });

    if (!response.ok) {
      const errorData = await response.text();
      await handleError(errorData);
      return;
    }

    return await response.json();
  } catch (error) {
    console.error("Error creating tag:", error);
    throw error;
  }
};

export const tagUpdateCall = async (accessToken: string, formValues: TagUpdateRequest): Promise<void> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/update` : `/tag/update`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(formValues),
    });

    if (!response.ok) {
      const errorData = await response.text();
      await handleError(errorData);
      return;
    }

    return await response.json();
  } catch (error) {
    console.error("Error updating tag:", error);
    throw error;
  }
};

export const tagInfoCall = async (accessToken: string, tagNames: string[]): Promise<TagInfoResponse> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/info` : `/tag/info`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ names: tagNames }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      await handleError(errorData);
      return {};
    }

    const data = await response.json();
    return data as TagInfoResponse;
  } catch (error) {
    console.error("Error getting tag info:", error);
    throw error;
  }
};

const formatYmd = (value: Date): string => {
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, "0");
  const day = String(value.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

export const tagListCall = async (
  accessToken: string,
  startTime?: Date | null,
  endTime?: Date | null,
): Promise<TagListResponse> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/list` : `/tag/list`;

    if (startTime && endTime) {
      const params = new URLSearchParams({
        start_date: formatYmd(startTime),
        end_date: formatYmd(endTime),
      });
      url = `${url}?${params.toString()}`;
    }

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      await handleError(errorData);
      return {};
    }

    const data = await response.json();
    return data as TagListResponse;
  } catch (error) {
    console.error("Error listing tags:", error);
    throw error;
  }
};

export const tagDeleteCall = async (accessToken: string, tagName: string): Promise<void> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/delete` : `/tag/delete`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ name: tagName }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      await handleError(errorData);
      return;
    }

    return await response.json();
  } catch (error) {
    console.error("Error deleting tag:", error);
    throw error;
  }
};

export const getDefaultTeamSettings = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/get/default_team_settings`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to fetch default team settings:", error);
    throw error;
  }
};

export const updateDefaultTeamSettings = async (accessToken: string, settings: Record<string, any>) => {
  try {
    const data = await apiClient.patch(`/update/default_team_settings`, { accessToken, body: settings });
    return data;
  } catch (error) {
    console.error("Failed to update default team settings:", error);
    throw error;
  }
};

export const getTeamPermissionsCall = async (accessToken: string, teamId: string) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/permissions_list?team_id=${teamId}`
      : `/team/permissions_list?team_id=${teamId}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      console.error("Available permissions fetch failed:", errorMessage);
      return { all_available_permissions: [], team_member_permissions: [] };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get team permissions:", error);
    throw error;
  }
};

export const teamPermissionsUpdateCall = async (accessToken: string, teamId: string, permissions: string[]) => {
  try {
    const data = await apiClient.post(`/team/permissions_update`, {
      accessToken,
      body: {
        team_id: teamId,
        team_member_permissions: permissions,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to update team permissions:", error);
    throw error;
  }
};

/**
 * Get a page of spend logs for a particular session.
 *
 * The backend paginates this endpoint (page / page_size, returning
 * { data, total, page, page_size, total_pages }). Callers that need the whole
 * session should page through total_pages and accumulate the results.
 */
export const sessionSpendLogsCall = async (
  accessToken: string,
  session_id: string,
  page: number = 1,
  page_size: number = 100,
) => {
  try {
    const params = new URLSearchParams({
      session_id,
      page: String(page),
      page_size: String(page_size),
    });
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/spend/logs/session/ui?${params.toString()}`
      : `/spend/logs/session/ui?${params.toString()}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch session logs:", error);
    throw error;
  }
};

export const vectorStoreCreateCall = async (accessToken: string, formValues: Record<string, any>): Promise<void> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/vector_store/new` : `/vector_store/new`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(formValues),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to create vector store");
    }

    return await response.json();
  } catch (error) {
    console.error("Error creating vector store:", error);
    throw error;
  }
};

export const vectorStoreListCall = async (
  accessToken: string,
  page: number = 1,
  page_size: number = 100,
): Promise<any> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/vector_store/list` : `/vector_store/list`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to list vector stores");
    }

    return await response.json();
  } catch (error) {
    console.error("Error listing vector stores:", error);
    throw error;
  }
};

export const vectorStoreDeleteCall = async (accessToken: string, vectorStoreId: string): Promise<void> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/vector_store/delete` : `/vector_store/delete`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ vector_store_id: vectorStoreId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to delete vector store");
    }

    return await response.json();
  } catch (error) {
    console.error("Error deleting vector store:", error);
    throw error;
  }
};

export const vectorStoreInfoCall = async (accessToken: string, vectorStoreId: string): Promise<any> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/vector_store/info` : `/vector_store/info`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ vector_store_id: vectorStoreId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to get vector store info");
    }

    return await response.json();
  } catch (error) {
    console.error("Error getting vector store info:", error);
    throw error;
  }
};

export const vectorStoreUpdateCall = async (accessToken: string, formValues: Record<string, any>): Promise<any> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/vector_store/update` : `/vector_store/update`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(formValues),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to update vector store");
    }

    return await response.json();
  } catch (error) {
    console.error("Error updating vector store:", error);
    throw error;
  }
};

export const ragIngestCall = async (
  accessToken: string,
  file: File,
  customLlmProvider: string,
  vectorStoreId?: string,
  vectorStoreName?: string,
  vectorStoreDescription?: string,
  providerSpecificParams?: Record<string, any>,
): Promise<any> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/rag/ingest` : `/rag/ingest`;

    const formData = new FormData();
    formData.append("file", file);

    const ingestOptions: any = {
      ingest_options: {
        vector_store: {
          custom_llm_provider: customLlmProvider,
          ...(vectorStoreId && { vector_store_id: vectorStoreId }),
          ...(providerSpecificParams && providerSpecificParams),
        },
      },
    };

    // Add litellm_vector_store_params if name or description provided
    if (vectorStoreName || vectorStoreDescription) {
      ingestOptions.ingest_options.litellm_vector_store_params = {};
      if (vectorStoreName) {
        ingestOptions.ingest_options.litellm_vector_store_params.vector_store_name = vectorStoreName;
      }
      if (vectorStoreDescription) {
        ingestOptions.ingest_options.litellm_vector_store_params.vector_store_description = vectorStoreDescription;
      }
    }

    formData.append("request", JSON.stringify(ingestOptions));

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error?.message || error.detail || "Failed to ingest document");
    }

    return await response.json();
  } catch (error) {
    console.error("Error ingesting document:", error);
    throw error;
  }
};

export const getEmailEventSettings = async (accessToken: string): Promise<EmailEventSettingsResponse> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/email/event_settings` : `/email/event_settings`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to get email event settings");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get email event settings:", error);
    throw error;
  }
};

export const updateEmailEventSettings = async (accessToken: string, settings: EmailEventSettingsUpdateRequest) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/email/event_settings` : `/email/event_settings`;

    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(settings),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to update email event settings");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to update email event settings:", error);
    throw error;
  }
};

export const resetEmailEventSettings = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/email/event_settings/reset` : `/email/event_settings/reset`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to reset email event settings");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to reset email event settings:", error);
    throw error;
  }
};

export { type Team } from "./key_team_helpers/key_list"; // Re-export Team

export const deleteAgentCall = async (accessToken: string, agentId: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/agents/${agentId}` : `/v1/agents/${agentId}`;

    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to delete agent:", error);
    throw error;
  }
};

export const makeAgentsPublicCall = async (accessToken: string, agentIds: string[]) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/agents/make_public` : `/v1/agents/make_public`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        agent_ids: agentIds,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to make agents public:", error);
    throw error;
  }
};

export const makeMCPPublicCall = async (accessToken: string, mcpServerIds: string[]) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/mcp/make_public` : `/v1/mcp/make_public`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        mcp_server_ids: mcpServerIds,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to make agents public:", error);
    throw error;
  }
};

export const deleteGuardrailCall = async (accessToken: string, guardrailId: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/${guardrailId}` : `/guardrails/${guardrailId}`;

    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error(errorData);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to delete guardrail:", error);
    throw error;
  }
};

export const getGuardrailUISettings = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/guardrails/ui/add_guardrail_settings`
      : `/guardrails/ui/add_guardrail_settings`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to get guardrail UI settings");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get guardrail UI settings:", error);
    throw error;
  }
};

export const getGuardrailProviderSpecificParams = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/guardrails/ui/provider_specific_params`
      : `/guardrails/ui/provider_specific_params`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to get guardrail provider specific parameters");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get guardrail provider specific parameters:", error);
    throw error;
  }
};

export const getCategoryYaml = async (accessToken: string, categoryName: string) => {
  try {
    // URL encode the category name to handle special characters
    const encodedCategoryName = encodeURIComponent(categoryName);
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/guardrails/ui/category_yaml/${encodedCategoryName}`
      : `/guardrails/ui/category_yaml/${encodedCategoryName}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error(`Failed to get category YAML. Status: ${response.status}, Error:`, errorData);
      handleError(errorData);
      throw new Error(`Failed to get category YAML: ${response.status} ${errorData}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get category YAML:", error);
    throw error;
  }
};

export const getMajorAirlines = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/ui/major_airlines` : `/guardrails/ui/major_airlines`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error(`Failed to get major airlines. Status: ${response.status}, Error:`, errorData);
      handleError(errorData);
      throw new Error(`Failed to get major airlines: ${response.status} ${errorData}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get major airlines:", error);
    throw error;
  }
};

export const getAgentsList = async (accessToken: string, healthCheck: boolean = false) => {
  try {
    const params = healthCheck ? "?health_check=true" : "";
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/agents${params}` : `/v1/agents${params}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to get agents list");
    }

    const data = await response.json();
    return { agents: data };
  } catch (error) {
    console.error("Failed to get agents list:", error);
    throw error;
  }
};

export const getAgentInfo = async (accessToken: string, agentId: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/agents/${agentId}` : `/v1/agents/${agentId}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to get agent info");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get agent info:", error);
    throw error;
  }
};

export const getGuardrailInfo = async (accessToken: string, guardrailId: string) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/${guardrailId}/info` : `/guardrails/${guardrailId}/info`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to get guardrail info");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to get guardrail info:", error);
    throw error;
  }
};

export const patchAgentCall = async (
  accessToken: string,
  agentId: string,
  updateData: {
    agent_name?: string;
    litellm_params?: Record<string, any>;
    agent_card_params?: Record<string, any>;
    tpm_limit?: number | null;
    rpm_limit?: number | null;
    session_tpm_limit?: number | null;
    session_rpm_limit?: number | null;
  },
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/agents/${agentId}` : `/v1/agents/${agentId}`;

    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updateData),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to patch agent");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to update guardrail:", error);
    throw error;
  }
};

export const updateGuardrailCall = async (
  accessToken: string,
  guardrailId: string,
  updateData: {
    guardrail_name?: string;
    default_on?: boolean;
    guardrail_info?: Record<string, any>;
    litellm_params?: Record<string, any>;
  },
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/${guardrailId}` : `/guardrails/${guardrailId}`;

    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updateData),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to update guardrail");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to update guardrail:", error);
    throw error;
  }
};

export const applyGuardrail = async (
  accessToken: string,
  guardrailName: string,
  text: string,
  language?: string | null,
  entities?: string[] | null,
  metadata?: Record<string, unknown> | null,
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/apply_guardrail` : `/guardrails/apply_guardrail`;

    const requestBody: Record<string, any> = {
      guardrail_name: guardrailName,
      text: text,
    };

    if (language) {
      requestBody.language = language;
    }

    if (entities && entities.length > 0) {
      requestBody.entities = entities;
    }

    if (metadata != null) {
      requestBody.metadata = metadata;
    }

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.text();
      let errorMessage = "Failed to apply guardrail";

      try {
        const errorJson = JSON.parse(errorData);
        if (errorJson.error?.message) {
          errorMessage = errorJson.error.message;
        } else if (errorJson.detail) {
          errorMessage = errorJson.detail;
        } else if (errorJson.message) {
          errorMessage = errorJson.message;
        }
      } catch (e) {
        errorMessage = errorData || errorMessage;
      }

      handleError(errorData);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to apply guardrail:", error);
    throw error;
  }
};

interface TestCustomCodeGuardrailRequest {
  custom_code: string;
  test_input: {
    texts: string[];
    images?: string[];
    tools?: Record<string, any>[];
    tool_calls?: Record<string, any>[];
    structured_messages?: Record<string, any>[];
    model?: string;
  };
  input_type?: "request" | "response";
  request_data?: {
    model?: string;
    user_id?: string;
    team_id?: string;
    end_user_id?: string;
    metadata?: Record<string, any>;
  };
}

interface TestCustomCodeGuardrailResponse {
  success: boolean;
  result?: {
    action: "allow" | "block" | "modify";
    reason?: string;
    texts?: string[];
    images?: string[];
    tool_calls?: Record<string, any>[];
    detection_info?: Record<string, any>;
    warning?: string;
  };
  error?: string;
  error_type?: "compilation" | "execution";
}

export const testCustomCodeGuardrail = async (
  accessToken: string,
  request: TestCustomCodeGuardrailRequest,
): Promise<TestCustomCodeGuardrailResponse> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/test_custom_code` : `/guardrails/test_custom_code`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.text();
      let errorMessage = "Failed to test custom code guardrail";

      try {
        const errorJson = JSON.parse(errorData);
        if (errorJson.error?.message) {
          errorMessage = errorJson.error.message;
        } else if (errorJson.detail) {
          errorMessage = errorJson.detail;
        } else if (errorJson.message) {
          errorMessage = errorJson.message;
        }
      } catch (e) {
        errorMessage = errorData || errorMessage;
      }

      handleError(errorData);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to test custom code guardrail:", error);
    throw error;
  }
};

export const validateBlockedWordsFile = async (accessToken: string, fileContent: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/guardrails/validate_blocked_words_file`
      : `/guardrails/validate_blocked_words_file`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ file_content: fileContent }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Failed to validate blocked words file");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to validate blocked words file:", error);
    throw error;
  }
};

export const getSSOSettings = async (accessToken: string) => {
  try {
    const data = await apiClient.get(`/get/sso_settings`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to fetch SSO configuration:", error);
    throw error;
  }
};

export const updateSSOSettings = async (accessToken: string, settings: Record<string, any>) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl ? `${proxyBaseUrl}/update/sso_settings` : `/update/sso_settings`;

    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(settings),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const detailMessage =
        typeof errorData?.detail === "object"
          ? errorData.detail?.error || errorData.detail?.message
          : errorData?.detail;
      const errorMessage =
        typeof detailMessage === "string" && detailMessage.length > 0 ? detailMessage : deriveErrorMessage(errorData);

      handleError(errorMessage);

      const enhancedError = new Error(errorMessage);
      if (errorData?.detail !== undefined) {
        (enhancedError as any).detail = errorData.detail;
      }
      (enhancedError as any).rawError = errorData;

      throw enhancedError;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to update SSO configuration:", error);
    throw error;
  }
};

interface UiAuditLogsParams {
  action?: string;
  table_name?: string;
  object_id?: string;
  changed_by?: string;
  changed_by_api_key?: string;
  object_team_id?: string;
  object_key_hash?: string;
  sort_by?: string;
  sort_order?: "asc" | "desc";
}

interface UiAuditLogsCallOptions {
  accessToken: string;
  page?: number;
  page_size?: number;
  params?: UiAuditLogsParams;
}

export const uiAuditLogsCall = async ({
  accessToken,
  page = 1,
  page_size = 50,
  params = {},
}: UiAuditLogsCallOptions) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/audit` : `/audit`;

    const queryParams = new URLSearchParams();
    queryParams.append("page", page.toString());
    queryParams.append("page_size", page_size.toString());

    for (const [key, value] of Object.entries(params)) {
      if (value != null && value !== "") {
        queryParams.append(key, String(value));
      }
    }

    url += `?${queryParams.toString()}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to fetch audit logs:", error);
    throw error;
  }
};

export const getRemainingUsers = async (
  accessToken: string,
): Promise<{
  total_users: number | null;
  total_users_used: number;
  total_users_remaining: number | null;
  total_teams: number | null;
  total_teams_used: number;
  total_teams_remaining: number | null;
} | null> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/available_users` : `/user/available_users`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      // if 404 - return None
      if (response.status === 404) {
        return null;
      }
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch remaining users:", error);
    throw error;
  }
};

export interface LicenseInfo {
  has_license: boolean;
  license_type: string | null;
  expiration_date: string | null;
  allowed_features: string[];
  limits: {
    max_users: number | null;
    max_teams: number | null;
  };
}

export const getLicenseInfo = async (accessToken: string): Promise<LicenseInfo | null> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/health/license` : `/health/license`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      // if 404 - return null (endpoint not available)
      if (response.status === 404) {
        return null;
      }
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch license info:", error);
    throw error;
  }
};

export const updatePassThroughEndpoint = async (
  accessToken: string,
  endpointPath: string,
  formValues: Record<string, any>,
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/pass_through_endpoint/${encodeURIComponent(endpointPath)}`
      : `/config/pass_through_endpoint/${encodeURIComponent(endpointPath)}`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formValues),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    NotificationsManager.success("Pass through endpoint updated successfully");
    return data;
  } catch (error) {
    console.error("Failed to update pass through endpoint:", error);
    throw error;
  }
};

export const deleteCallback = async (accessToken: string, callbackName: string) => {
  /**
   * Delete specific callback from proxy using the /config/callback/delete API
   */
  try {
    const data = await apiClient.post(`/config/callback/delete`, {
      accessToken,
      body: {
        callback_name: callbackName,
      },
    });
    return data;
  } catch (error) {
    console.error("Failed to delete specific callback:", error);
    throw error;
  }
};

export const testMCPToolsListRequest = async (
  accessToken: string | null,
  mcpServerConfig: Record<string, any>,
  oauthAccessToken?: string | null,
) => {
  try {
    // Construct the URL for POST request
    const url = proxyBaseUrl ? `${proxyBaseUrl}/mcp-rest/test/tools/list` : `/mcp-rest/test/tools/list`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (accessToken) {
      headers["x-litellm-api-key"] = accessToken;
      if (globalLitellmHeaderName.toLowerCase() !== "authorization") {
        headers[globalLitellmHeaderName] = `Bearer ${accessToken}`;
      }
    }
    if (oauthAccessToken) {
      headers["Authorization"] = `Bearer ${oauthAccessToken}`;
    } else if (accessToken) {
      headers[globalLitellmHeaderName] = `Bearer ${accessToken}`;
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(mcpServerConfig),
    });

    // Check for non-JSON responses first
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const text = await response.text();
      console.error("Received non-JSON response:", text);
      throw new Error(
        `Received non-JSON response (${response.status}: ${response.statusText}). Check network tab for details.`,
      );
    }

    const data = await response.json();

    if (!response.ok || data.error) {
      if (response.status === 403) {
        return {
          tools: [],
          error: true,
          status: 403,
          message: MCP_TOOLS_PREVIEW_FORBIDDEN_MESSAGE,
        };
      }
      // Return the error response instead of throwing an error
      // This allows the caller to handle the error format properly
      if (data.error) {
        return { ...data, status: response.status };
      }
      return {
        tools: [],
        error: "request_failed",
        status: response.status,
        message: data.message || `MCP tools list failed: ${response.status} ${response.statusText}`,
      };
    }

    return data;
  } catch (error) {
    console.error("MCP tools list test error:", error);
    // For network errors or other exceptions, still throw
    throw error;
  }
};

export const cacheTemporaryMcpServer = async (accessToken: string, payload: Record<string, any>) => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/mcp/server/oauth/session` : `/v1/mcp/server/oauth/session`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    const errorMessage = deriveErrorMessage(data) || data?.error || "Failed to cache MCP server";
    throw new Error(errorMessage);
  }
  return data;
};

interface RegisterMcpOAuthClientPayload {
  client_name?: string;
  grant_types?: string[];
  response_types?: string[];
  token_endpoint_auth_method?: string;
  redirect_uris?: string[];
}

export const registerMcpOAuthClient = async (
  accessToken: string,
  serverId: string,
  payload: RegisterMcpOAuthClientPayload,
) => {
  const base = getProxyBaseUrl();
  const normalizedServerId = encodeURIComponent(serverId.trim());
  const url = `${base}/v1/mcp/server/oauth/${normalizedServerId}/register`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    const errorMessage = deriveErrorMessage(data) || data?.detail || "Failed to register OAuth client";
    throw new Error(errorMessage);
  }
  return data;
};

interface BuildOAuthAuthorizeURLParams {
  serverId: string;
  clientId?: string;
  redirectUri: string;
  state: string;
  codeChallenge: string;
  scope?: string;
}

export const buildMcpOAuthAuthorizeUrl = ({
  serverId,
  clientId,
  redirectUri,
  state,
  codeChallenge,
  scope,
}: BuildOAuthAuthorizeURLParams): string => {
  const base = getProxyBaseUrl();
  const normalizedServerId = encodeURIComponent(serverId.trim());
  const url = `${base}/v1/mcp/server/oauth/${normalizedServerId}/authorize`;
  const params = new URLSearchParams({
    redirect_uri: redirectUri,
    state,
    response_type: "code",
    code_challenge: codeChallenge,
    code_challenge_method: "S256",
  });
  if (clientId && clientId.trim().length > 0) {
    params.set("client_id", clientId);
  }
  if (scope && scope.trim().length > 0) {
    params.set("scope", scope);
  }
  return `${url}?${params.toString()}`;
};

interface ExchangeMcpOAuthTokenParams {
  serverId: string;
  code: string;
  clientId?: string;
  clientSecret?: string;
  codeVerifier: string;
  redirectUri: string;
  accessToken?: string | null;
}

export const exchangeMcpOAuthToken = async ({
  serverId,
  code,
  clientId,
  clientSecret,
  codeVerifier,
  redirectUri,
  accessToken,
}: ExchangeMcpOAuthTokenParams) => {
  const base = getProxyBaseUrl();
  const normalizedServerId = encodeURIComponent(serverId.trim());
  const url = `${base}/v1/mcp/server/oauth/${normalizedServerId}/token`;

  const body = new URLSearchParams();
  body.set("grant_type", "authorization_code");
  body.set("code", code);
  if (clientId && clientId.trim().length > 0) {
    body.set("client_id", clientId);
  }
  if (clientSecret && clientSecret.trim().length > 0) {
    body.set("client_secret", clientSecret);
  }
  body.set("code_verifier", codeVerifier);
  body.set("redirect_uri", redirectUri);

  const headers: Record<string, string> = {
    "Content-Type": "application/x-www-form-urlencoded",
  };
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: body.toString(),
  });

  const data = await response.json();
  if (!response.ok) {
    const oauthErrorMessage =
      typeof data?.error === "string" && typeof data?.error_description === "string"
        ? `${data.error}: ${data.error_description}`
        : undefined;
    const errorMessage = oauthErrorMessage || deriveErrorMessage(data) || data?.detail || "OAuth token exchange failed";
    throw new Error(errorMessage);
  }
  return data;
};

export const vectorStoreSearchCall = async (
  accessToken: string,
  vectorStoreId: string,
  query: string,
): Promise<any> => {
  try {
    const url = `${getProxyBaseUrl()}/v1/vector_stores/${vectorStoreId}/search`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      await handleError(errorData);
      return null;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error testing vector store search:", error);
    throw error;
  }
};

export const searchToolQueryCall = async (
  accessToken: string,
  searchToolName: string,
  query: string,
  maxResults?: number,
): Promise<any> => {
  try {
    const url = `${getProxyBaseUrl()}/v1/search/${searchToolName}`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        max_results: maxResults || 5,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      await handleError(errorData);
      return null;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error querying search tool:", error);
    throw error;
  }
};

// New endpoint functions for DAU, WAU, MAU
export const tagDauCall = async (accessToken: string, endDate: Date, tagFilter?: string, tagFilters?: string[]) => {
  /**
   * Get Daily Active Users (DAU) for last 7 days ending on endDate
   */
  try {
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    const hasTagFilters = tagFilters && tagFilters.length > 0;
    return await apiClient.get(`/tag/dau`, {
      accessToken,
      query: {
        end_date: formatDate(endDate),
        tag_filters: hasTagFilters ? tagFilters : undefined,
        tag_filter: !hasTagFilters && tagFilter ? tagFilter : undefined,
      },
    });
  } catch (error) {
    console.error("Failed to fetch DAU:", error);
    throw error;
  }
};

export const tagWauCall = async (accessToken: string, endDate: Date, tagFilter?: string, tagFilters?: string[]) => {
  /**
   * Get Weekly Active Users (WAU) for last 7 weeks ending on endDate
   */
  try {
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    const hasTagFilters = tagFilters && tagFilters.length > 0;
    return await apiClient.get(`/tag/wau`, {
      accessToken,
      query: {
        end_date: formatDate(endDate),
        tag_filters: hasTagFilters ? tagFilters : undefined,
        tag_filter: !hasTagFilters && tagFilter ? tagFilter : undefined,
      },
    });
  } catch (error) {
    console.error("Failed to fetch WAU:", error);
    throw error;
  }
};

export const tagMauCall = async (accessToken: string, endDate: Date, tagFilter?: string, tagFilters?: string[]) => {
  /**
   * Get Monthly Active Users (MAU) for last 7 months ending on endDate
   */
  try {
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    const hasTagFilters = tagFilters && tagFilters.length > 0;
    return await apiClient.get(`/tag/mau`, {
      accessToken,
      query: {
        end_date: formatDate(endDate),
        tag_filters: hasTagFilters ? tagFilters : undefined,
        tag_filter: !hasTagFilters && tagFilter ? tagFilter : undefined,
      },
    });
  } catch (error) {
    console.error("Failed to fetch MAU:", error);
    throw error;
  }
};

export const tagDistinctCall = async (accessToken: string) => {
  /**
   * Get all distinct user agent tags (up to 250)
   */
  try {
    const data = await apiClient.get(`/tag/distinct`, { accessToken });
    return data;
  } catch (error) {
    console.error("Failed to fetch distinct tags:", error);
    throw error;
  }
};

export const userAgentSummaryCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  tagFilters?: string[],
) => {
  /**
   * Get user agent summary statistics
   */
  try {
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    return await apiClient.get(`/tag/summary`, {
      accessToken,
      query: {
        start_date: formatDate(startTime),
        end_date: formatDate(endTime),
        tag_filters: tagFilters && tagFilters.length > 0 ? tagFilters : undefined,
      },
    });
  } catch (error) {
    console.error("Failed to fetch user agent summary:", error);
    throw error;
  }
};

export const perUserAnalyticsCall = async (
  accessToken: string,
  page: number = 1,
  pageSize: number = 50,
  tagFilters?: string[],
) => {
  /**
   * Get per-user analytics data for the last 30 days
   */
  try {
    return await apiClient.get(`/tag/user-agent/per-user-analytics`, {
      accessToken,
      query: {
        page: page.toString(),
        page_size: pageSize.toString(),
        tag_filters: tagFilters && tagFilters.length > 0 ? tagFilters : undefined,
      },
    });
  } catch (error) {
    console.error("Failed to fetch per-user analytics:", error);
    throw error;
  }
};

export interface LoginRequest {
  username: string;
  password: string;
  useV3?: boolean;
}

interface LoginResponse {
  redirect_url: string;
  token?: string;
  code?: string;
  expires_in?: number;
}

export const loginCall = async (username: string, password: string, useV3?: boolean): Promise<LoginResponse> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const loginPath = useV3 ? "/v3/login" : "/v2/login";
  const loginUrl = proxyBaseUrl ? `${proxyBaseUrl}${loginPath}` : loginPath;

  const body = JSON.stringify({
    username,
    password,
  });

  const response = await fetch(loginUrl, {
    method: "POST",
    body,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    throw new Error(errorMessage);
  }

  const data: LoginResponse = await response.json();

  // v3 returns an opaque code — exchange it for the real JWT
  if (useV3 && data.code) {
    const exchangeUrl = proxyBaseUrl ? `${proxyBaseUrl}/v3/login/exchange` : "/v3/login/exchange";

    const exchangeResponse = await fetch(exchangeUrl, {
      method: "POST",
      body: JSON.stringify({ code: data.code }),
      credentials: "include",
      headers: { "Content-Type": "application/json" },
    });

    if (!exchangeResponse.ok) {
      const errorData = await exchangeResponse.json();
      throw new Error(deriveErrorMessage(errorData));
    }

    const exchangeData: LoginResponse = await exchangeResponse.json();
    if (exchangeData.token) {
      storeLoginToken(exchangeData.token);
    }
    return exchangeData;
  }

  // Backwards compatibility: v2 or old v3 returns token directly
  if (data.token) {
    storeLoginToken(data.token);
  }

  return data;
};

/**
 * Exchange a single-use login code for a JWT token.
 * Used by the SSO callback when the worker redirects back with ?code=.
 */
export const exchangeLoginCode = async (code: string, workerBaseUrl?: string | null): Promise<string> => {
  const base = workerBaseUrl || getProxyBaseUrl();
  const response = await fetch(`${base}/v3/login/exchange`, {
    method: "POST",
    body: JSON.stringify({ code }),
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(deriveErrorMessage(errorData));
  }

  const data = await response.json();
  if (data.token) {
    document.cookie = `token=${data.token}; path=/; SameSite=Lax`;
  }
  return data.token;
};

export const getUiSettings = async () => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/get/ui_settings` : `/get/ui_settings`;
  const response = await fetch(url, {
    method: "GET",
  });
  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    throw new Error(errorMessage);
  }
  const data = await response.json();
  return data;
};

export const updateUiSettings = async (accessToken: string, settings: Record<string, any>) => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl ? `${proxyBaseUrl}/update/ui_settings` : `/update/ui_settings`;
  const response = await fetch(url, {
    method: "PATCH",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(settings),
  });
  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    throw new Error(errorMessage);
  }
  const data = await response.json();
  return data;
};

// Claude Code Marketplace Networking Functions

/**
 * Get public marketplace catalog (no authentication required)
 * Returns marketplace.json for Claude Code CLI discovery
 */
export const getClaudeCodeMarketplace = async () => {
  try {
    const proxyBaseUrl = getProxyBaseUrl();
    const url = proxyBaseUrl ? `${proxyBaseUrl}/claude-code/marketplace.json` : `/claude-code/marketplace.json`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      const errorMessage = deriveErrorMessage(JSON.parse(errorData));
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch Claude Code marketplace:", error);
    throw error;
  }
};

/**
 * List all Claude Code plugins (admin only)
 * @param accessToken - Admin access token
 * @param enabledOnly - If true, only return enabled plugins (default: false)
 */
export const getClaudeCodePluginsList = async (accessToken: string, enabledOnly: boolean = false) => {
  try {
    const proxyBaseUrl = getProxyBaseUrl();
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/claude-code/plugins?enabled_only=${enabledOnly}`
      : `/claude-code/plugins?enabled_only=${enabledOnly}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      const errorMessage = deriveErrorMessage(JSON.parse(errorData));
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch Claude Code plugins list:", error);
    throw error;
  }
};

/**
 * Get details for a specific Claude Code plugin (admin only)
 * @param accessToken - Admin access token
 * @param pluginName - Name of the plugin
 */
export const getClaudeCodePluginDetails = async (accessToken: string, pluginName: string) => {
  try {
    const proxyBaseUrl = getProxyBaseUrl();
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/claude-code/plugins/${pluginName}`
      : `/claude-code/plugins/${pluginName}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      const errorMessage = deriveErrorMessage(JSON.parse(errorData));
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to fetch plugin "${pluginName}":`, error);
    throw error;
  }
};

/**
 * Register or update a Claude Code plugin (admin only)
 * @param accessToken - Admin access token
 * @param pluginData - Plugin registration data
 */
export const registerClaudeCodePlugin = async (accessToken: string, pluginData: SkillRegisterRequest) => {
  try {
    const proxyBaseUrl = getProxyBaseUrl();
    const url = proxyBaseUrl ? `${proxyBaseUrl}/claude-code/plugins` : `/claude-code/plugins`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(pluginData),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      let errorMessage: string;
      try {
        errorMessage = deriveErrorMessage(JSON.parse(errorBody));
      } catch {
        errorMessage = errorBody || `Request failed with status ${response.status}`;
      }
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to register Claude Code plugin:", error);
    throw error;
  }
};

/**
 * Enable a Claude Code plugin (admin only)
 * @param accessToken - Admin access token
 * @param pluginName - Name of the plugin to enable
 */
export const enableClaudeCodePlugin = async (accessToken: string, pluginName: string) => {
  try {
    const proxyBaseUrl = getProxyBaseUrl();
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/claude-code/plugins/${pluginName}/enable`
      : `/claude-code/plugins/${pluginName}/enable`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      const errorMessage = deriveErrorMessage(JSON.parse(errorData));
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to enable plugin "${pluginName}":`, error);
    throw error;
  }
};

/**
 * Disable a Claude Code plugin (admin only)
 * @param accessToken - Admin access token
 * @param pluginName - Name of the plugin to disable
 */
export const disableClaudeCodePlugin = async (accessToken: string, pluginName: string) => {
  try {
    const proxyBaseUrl = getProxyBaseUrl();
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/claude-code/plugins/${pluginName}/disable`
      : `/claude-code/plugins/${pluginName}/disable`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      const errorMessage = deriveErrorMessage(JSON.parse(errorData));
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to disable plugin "${pluginName}":`, error);
    throw error;
  }
};

/**
 * Delete a Claude Code plugin (admin only)
 * @param accessToken - Admin access token
 * @param pluginName - Name of the plugin to delete
 */
export const deleteClaudeCodePlugin = async (accessToken: string, pluginName: string) => {
  try {
    const proxyBaseUrl = getProxyBaseUrl();
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/claude-code/plugins/${pluginName}`
      : `/claude-code/plugins/${pluginName}`;

    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      const errorMessage = deriveErrorMessage(JSON.parse(errorData));
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to delete plugin "${pluginName}":`, error);
    throw error;
  }
};

// Compliance check types and functions

export interface ComplianceCheckResult {
  check_name: string;
  article: string;
  passed: boolean;
  detail: string;
}

export interface ComplianceResponse {
  compliant: boolean;
  regulation: string;
  checks: ComplianceCheckResult[];
}

export interface ComplianceCheckRequest {
  request_id: string;
  user_id?: string;
  model?: string;
  timestamp?: string;
  guardrail_information?: Record<string, any>[];
}

export const checkEuAiActCompliance = async (
  accessToken: string,
  payload: ComplianceCheckRequest,
): Promise<ComplianceResponse> => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/compliance/eu-ai-act` : `/compliance/eu-ai-act`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

export const checkGdprCompliance = async (
  accessToken: string,
  payload: ComplianceCheckRequest,
): Promise<ComplianceResponse> => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/compliance/gdpr` : `/compliance/gdpr`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

export interface ToolRow {
  tool_id: string;
  tool_name: string;
  origin?: string;
  input_policy: string;
  output_policy: string;
  call_count?: number;
  assignments?: Record<string, any>;
  key_hash?: string;
  team_id?: string;
  key_alias?: string;
  created_at?: string;
  updated_at?: string;
  created_by?: string;
  updated_by?: string;
  user_agent?: string;
  last_used_at?: string;
}

export interface ToolPolicyOption {
  value: string;
  label: string;
  description: string;
}

export interface ToolPolicyOptionsResponse {
  input_policies: ToolPolicyOption[];
  output_policies: ToolPolicyOption[];
}

export const fetchToolPolicyOptions = async (accessToken: string): Promise<ToolPolicyOptionsResponse> => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/tool/policy/options` : `/v1/tool/policy/options`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

export const fetchToolsList = async (accessToken: string): Promise<ToolRow[]> => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/tool/list` : `/v1/tool/list`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  const data = await response.json();
  return data.tools ?? [];
};

export interface ToolPolicyOverrideRow {
  override_id: string;
  tool_name: string;
  team_id?: string | null;
  key_hash?: string | null;
  input_policy: string;
  key_alias?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface ToolDetailResponse {
  tool: ToolRow;
  overrides: ToolPolicyOverrideRow[];
}

export interface ToolUsageLogEntry {
  id: string;
  timestamp: string;
  model?: string | null;
  spend?: number | null;
  total_tokens?: number | null;
  input_snippet?: string | null;
}

export interface ToolUsageLogsResponse {
  logs: ToolUsageLogEntry[];
  total: number;
  page: number;
  page_size: number;
}

export const getToolUsageLogs = async (
  accessToken: string,
  toolName: string,
  options: { page?: number; pageSize?: number; startDate?: string; endDate?: string },
): Promise<ToolUsageLogsResponse> => {
  const encoded = encodeURIComponent(toolName);
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/tool/${encoded}/logs` : `/v1/tool/${encoded}/logs`;
  const params = new URLSearchParams();
  if (options.page != null) params.append("page", String(options.page));
  if (options.pageSize != null) params.append("page_size", String(options.pageSize));
  if (options.startDate) params.append("start_date", options.startDate);
  if (options.endDate) params.append("end_date", options.endDate);
  const fullUrl = params.toString() ? `${url}?${params.toString()}` : url;
  const response = await fetch(fullUrl, {
    method: "GET",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(deriveErrorMessage(errorData));
  }
  return response.json();
};

export const fetchToolDetail = async (accessToken: string, toolName: string): Promise<ToolDetailResponse> => {
  const encoded = encodeURIComponent(toolName);
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/tool/${encoded}/detail` : `/v1/tool/${encoded}/detail`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

export const updateToolPolicy = async (
  accessToken: string,
  toolName: string,
  policies: { input_policy?: string; output_policy?: string },
  options?: { team_id?: string | null; key_hash?: string | null; key_alias?: string | null },
): Promise<ToolRow> => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/tool/policy` : `/v1/tool/policy`;
  const body: Record<string, string | undefined | null> = {
    tool_name: toolName,
  };
  if (policies.input_policy != null) body.input_policy = policies.input_policy;
  if (policies.output_policy != null) body.output_policy = policies.output_policy;
  if (options?.team_id != null) body.team_id = options.team_id || undefined;
  if (options?.key_hash != null) body.key_hash = options.key_hash || undefined;
  if (options?.key_alias != null) body.key_alias = options.key_alias || undefined;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

export const deleteToolPolicyOverride = async (
  accessToken: string,
  toolName: string,
  params: { team_id?: string | null; key_hash?: string | null },
): Promise<{ deleted: boolean; tool_name: string }> => {
  const encoded = encodeURIComponent(toolName);
  const q = new URLSearchParams();
  if (params.team_id != null && params.team_id !== "") q.set("team_id", params.team_id);
  if (params.key_hash != null && params.key_hash !== "") q.set("key_hash", params.key_hash);
  const query = q.toString();
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/tool/${encoded}/overrides${query ? `?${query}` : ""}`
    : `/v1/tool/${encoded}/overrides${query ? `?${query}` : ""}`;
  const response = await fetch(url, {
    method: "DELETE",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

// ── MCP OAuth user-credential helpers ────────────────────────────────────────

export interface MCPOAuthUserCredentialStatus {
  server_id: string;
  has_credential: boolean;
  expires_at?: string | null;
  is_expired: boolean;
  connected_at?: string | null;
}

export interface MCPUserCredentialListItem {
  server_id: string;
  server_name?: string | null;
  alias?: string | null;
  credential_type: string;
  has_credential: boolean;
  expires_at?: string | null;
  connected_at?: string | null;
}

export const storeMCPOAuthUserCredential = async (
  accessToken: string,
  serverId: string,
  tokenResponse: { access_token: string; refresh_token?: string; expires_in?: number; scopes?: string[] },
): Promise<MCPOAuthUserCredentialStatus> => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/mcp/server/${serverId}/oauth-user-credential`
    : `/v1/mcp/server/${serverId}/oauth-user-credential`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(tokenResponse),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    const errObj = err as { detail?: unknown };
    const detail = errObj?.detail;
    const detailMsg = Array.isArray(detail)
      ? detail
          .map((d: unknown) =>
            d && typeof d === "object" ? (d as Record<string, unknown>).msg ?? JSON.stringify(d) : String(d),
          )
          .join("; ")
      : typeof detail === "string"
        ? detail
        : detail && typeof (detail as Record<string, unknown>).error === "string"
          ? ((detail as Record<string, unknown>).error as string)
          : undefined;
    throw new Error(detailMsg || "Failed to store OAuth credential");
  }
  return response.json();
};

export const deleteMCPOAuthUserCredential = async (
  accessToken: string,
  serverId: string,
): Promise<MCPOAuthUserCredentialStatus> => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/mcp/server/${serverId}/oauth-user-credential`
    : `/v1/mcp/server/${serverId}/oauth-user-credential`;
  const response = await fetch(url, {
    method: "DELETE",
    headers: { [globalLitellmHeaderName]: `Bearer ${accessToken}` },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    const errObj = err as { detail?: unknown };
    const detail = errObj?.detail;
    const detailMsg = Array.isArray(detail)
      ? detail
          .map((d: unknown) =>
            d && typeof d === "object" ? (d as Record<string, unknown>).msg ?? JSON.stringify(d) : String(d),
          )
          .join("; ")
      : typeof detail === "string"
        ? detail
        : detail && typeof (detail as Record<string, unknown>).error === "string"
          ? ((detail as Record<string, unknown>).error as string)
          : undefined;
    throw new Error(detailMsg || "Failed to revoke OAuth credential");
  }
  return response.json();
};

export const getMCPOAuthUserCredentialStatus = async (
  accessToken: string,
  serverId: string,
): Promise<MCPOAuthUserCredentialStatus> => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/mcp/server/${serverId}/oauth-user-credential/status`
    : `/v1/mcp/server/${serverId}/oauth-user-credential/status`;
  const response = await fetch(url, {
    method: "GET",
    headers: { [globalLitellmHeaderName]: `Bearer ${accessToken}` },
  });
  if (!response.ok) {
    return { server_id: serverId, has_credential: false, is_expired: false };
  }
  return response.json();
};

export const listMCPUserCredentials = async (accessToken: string): Promise<MCPUserCredentialListItem[]> => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/mcp/user-credentials` : `/v1/mcp/user-credentials`;
  const response = await fetch(url, {
    method: "GET",
    headers: { [globalLitellmHeaderName]: `Bearer ${accessToken}` },
  });
  if (!response.ok) return [];
  return response.json();
};

// ============================================================
// MCP per-user env vars (/v1/mcp/server/{id}/user-env-vars)
// ============================================================

export const getMCPUserEnvVars = async (accessToken: string, serverId: string): Promise<MCPUserEnvVarsStatus> => {
  return apiClient.get<MCPUserEnvVarsStatus>(`/v1/mcp/server/${serverId}/user-env-vars`, { accessToken });
};

export const storeMCPUserEnvVars = async (
  accessToken: string,
  serverId: string,
  values: Record<string, string>,
): Promise<MCPUserEnvVarsStatus> => {
  return apiClient.post<MCPUserEnvVarsStatus>(`/v1/mcp/server/${serverId}/user-env-vars`, {
    accessToken,
    body: { values },
  });
};

export const listMCPUserEnvVarStatus = async (accessToken: string): Promise<MCPUserEnvVarsStatus[]> => {
  // Best-effort status badges: a failure here must not break the page, so fall
  // back to an empty list rather than surfacing the error to the caller.
  try {
    return await apiClient.get<MCPUserEnvVarsStatus[]>("/v1/mcp/user-env-vars/status", { accessToken });
  } catch {
    return [];
  }
};

// ============================================================
// Memory management (/v1/memory)
// ============================================================

/**
 * Encode a memory key for use in a URL path segment.
 *
 * The backend route is declared as `/v1/memory/{key:path}`, which supports
 * slashes in the key (e.g. `user/123/notes`). Plain `encodeURIComponent`
 * encodes `/` as `%2F`, and some proxies/middlewares (nginx default,
 * CloudFlare, AWS ALB) either reject or silently re-decode `%2F`, which
 * can break the request before FastAPI ever sees it.
 *
 * We keep slashes literal as path delimiters while still encoding every
 * other potentially-unsafe character (spaces, `?`, `#`, `%`, etc.) per
 * path segment.
 */
const encodeMemoryKeyForPath = (key: string): string => key.split("/").map(encodeURIComponent).join("/");

export interface MemoryRow {
  memory_id: string;
  key: string;
  value: string;
  metadata?: unknown;
  user_id?: string | null;
  team_id?: string | null;
  created_at?: string;
  created_by?: string | null;
  updated_at?: string;
  updated_by?: string | null;
}

export interface MemoryListResponse {
  memories: MemoryRow[];
  total: number;
}

export const fetchMemoryList = async (
  accessToken: string,
  options: {
    key?: string;
    keyPrefix?: string;
    page?: number;
    pageSize?: number;
  } = {},
): Promise<MemoryListResponse> => {
  const base = proxyBaseUrl ? `${proxyBaseUrl}/v1/memory` : `/v1/memory`;
  const params = new URLSearchParams();
  // keyPrefix takes precedence — backend also does, but we omit `key`
  // to keep the URL clean and intent obvious.
  if (options.keyPrefix) {
    params.append("key_prefix", options.keyPrefix);
  } else if (options.key) {
    params.append("key", options.key);
  }
  if (options.page != null) params.append("page", String(options.page));
  if (options.pageSize != null) params.append("page_size", String(options.pageSize));
  const url = params.toString() ? `${base}?${params.toString()}` : base;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

export const createMemory = async (
  accessToken: string,
  payload: { key: string; value: string; metadata?: unknown },
): Promise<MemoryRow> => {
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/memory` : `/v1/memory`;
  const body: Record<string, unknown> = {
    key: payload.key,
    value: payload.value,
  };
  if (payload.metadata !== undefined) body.metadata = payload.metadata;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

export const updateMemory = async (
  accessToken: string,
  key: string,
  payload: { value?: string; metadata?: unknown },
): Promise<MemoryRow> => {
  const encoded = encodeMemoryKeyForPath(key);
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/memory/${encoded}` : `/v1/memory/${encoded}`;
  const response = await fetch(url, {
    method: "PUT",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
  return response.json();
};

export const deleteMemory = async (accessToken: string, key: string): Promise<void> => {
  const encoded = encodeMemoryKeyForPath(key);
  const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/memory/${encoded}` : `/v1/memory/${encoded}`;
  const response = await fetch(url, {
    method: "DELETE",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    throw new Error(errorData);
  }
};
