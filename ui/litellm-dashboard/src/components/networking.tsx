// Shared date formatter for daily activity endpoints
export const formatDate = (date: Date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};
/**
 * Helper file for calls being made to proxy
 */
import { all_admin_roles } from "@/utils/roles";
import { message } from "antd";
import {
  TagNewRequest,
  TagUpdateRequest,
  TagDeleteRequest,
  TagInfoRequest,
  TagListResponse,
  TagInfoResponse,
} from "./tag_management/types";
import { Team } from "./key_team_helpers/key_list";
import { UserInfo } from "./view_users/types";
import {
  EmailEventSettingsResponse,
  EmailEventSettingsUpdateRequest,
} from "./email_events/types";
import { jsonFields } from "./common_components/check_openapi_schema"
import NotificationsManager from "./molecules/notifications_manager";

const isLocal = process.env.NODE_ENV === "development";
export const defaultProxyBaseUrl = isLocal ? "http://localhost:4000" : null;
const defaultServerRootPath = "/";
export let serverRootPath = defaultServerRootPath;
export let proxyBaseUrl = defaultProxyBaseUrl;
if (isLocal != true) {
  console.log = function () {};
}

const updateProxyBaseUrl = (
  serverRootPath: string,
  receivedProxyBaseUrl: string | null = null
) => {
  /**
   * Special function for updating the proxy base url. Should only be called by getUiConfig.
   */
  const defaultProxyBaseUrl = isLocal
    ? "http://localhost:4000"
    : window.location.origin;
  let initialProxyBaseUrl = receivedProxyBaseUrl || defaultProxyBaseUrl;
  console.log("proxyBaseUrl:", proxyBaseUrl);
  console.log("serverRootPath:", serverRootPath);
  if (
    serverRootPath.length > 0 &&
    !initialProxyBaseUrl.endsWith(serverRootPath) &&
    serverRootPath != "/"
  ) {
    initialProxyBaseUrl += serverRootPath;
    proxyBaseUrl = initialProxyBaseUrl;
  }
  console.log("Updated proxyBaseUrl:", proxyBaseUrl);
};

const updateServerRootPath = (receivedServerRootPath: string) => {
  serverRootPath = receivedServerRootPath;
};

export const getProxyBaseUrl = (): string => {
  return proxyBaseUrl ? proxyBaseUrl : window.location.origin;
};

const HTTP_REQUEST = {
  GET: "GET",
  POST: "POST",
  PUT: "PUT",
  DELETE: "DELETE",
};

export const DEFAULT_ORGANIZATION = "default_organization";

export interface Model {
  model_name: string;
  litellm_params: Object;
  model_info: Object | null;
}

interface PromptInfo {
  prompt_type: string;
}

export interface PromptSpec {
  prompt_id: string;
  litellm_params: Object;
  prompt_info: PromptInfo;
  created_at?: string
  updated_at?: string
}

export interface PromptTemplateBase {
  litellm_prompt_id: string;
  content: string;
  metadata?: Record<string, any> | null;
}

export interface PromptInfoResponse {
  prompt_spec: PromptSpec;
  raw_prompt_template: PromptTemplateBase | null;
}

export interface ListPromptsResponse {
  prompts: PromptSpec[];
}

export interface Organization {
  organization_id: string | null;
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

export interface PublicModelHubInfo {
  docs_title: string;
  custom_docs_description: string | null;
  litellm_version: string;
  useful_links: Record<string, string>;
}

export interface LiteLLMWellKnownUiConfig {
  server_root_path: string;
  proxy_base_url: string | null;
}

export interface CredentialsResponse {
  credentials: CredentialItem[];
}

let lastErrorTime = 0;

const handleError = async (errorData: string) => {
  const currentTime = Date.now();
  if (currentTime - lastErrorTime > 60000) {
    // 60000 milliseconds = 60 seconds
    if (errorData.includes("Authentication Error - Expired Key")) {
      NotificationsManager.info("UI Session Expired. Logging out.");
      lastErrorTime = currentTime;
      document.cookie =
        "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
      window.location.href = window.location.pathname;
    }
    lastErrorTime = currentTime;
  } else {
    console.log("Error suppressed to prevent spam:", errorData);
  }
};

// Global variable for the header name
let globalLitellmHeaderName: string  = "Authorization";
const MCP_AUTH_HEADER: string  = "x-mcp-auth";

// Function to set the global header name
export function setGlobalLitellmHeaderName(
  headerName: string = "Authorization"
) {
  console.log(`setGlobalLitellmHeaderName: ${headerName}`);
  globalLitellmHeaderName = headerName;
}

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
  console.log("Getting UI config");
  /**Special route to get the proxy base url and server root path */
  const url = defaultProxyBaseUrl
    ? `${defaultProxyBaseUrl}/litellm/.well-known/litellm-ui-config`
    : `/litellm/.well-known/litellm-ui-config`;
  const response = await fetch(url);
  const jsonData: LiteLLMWellKnownUiConfig = await response.json();
  /**
   * Update the proxy base url and server root path
   */
  console.log("jsonData in getUiConfig:", jsonData);
  updateProxyBaseUrl(jsonData.server_root_path, jsonData.proxy_base_url);
  return jsonData;
};

export const getPublicModelHubInfo = async () => {
  const url = defaultProxyBaseUrl
    ? `${defaultProxyBaseUrl}/public/model_hub/info`
    : `/public/model_hub/info`;
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

export const modelCostMap = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/get/litellm_model_cost_map`
      : `/get/litellm_model_cost_map`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    const jsonData = await response.json();
    console.log(`received litellm model cost data: ${jsonData}`);
    return jsonData;
  } catch (error) {
    console.error("Failed to get model cost map:", error);
    throw error;
  }
};

export const reloadModelCostMap = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/reload/model_cost_map`
      : `/reload/model_cost_map`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    const jsonData = await response.json();
    console.log(`Model cost map reload response: ${jsonData}`);
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
    console.log(`Schedule model cost map reload response: ${jsonData}`);
    return jsonData;
  } catch (error) {
    console.error("Failed to schedule model cost map reload:", error);
    throw error;
  }
};

export const cancelModelCostMapReload = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/schedule/model_cost_map_reload`
      : `/schedule/model_cost_map_reload`;
    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    const jsonData = await response.json();
    console.log(`Cancel model cost map reload response: ${jsonData}`);
    return jsonData;
  } catch (error) {
    console.error("Failed to cancel model cost map reload:", error);
    throw error;
  }
};

export const getModelCostMapReloadStatus = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/schedule/model_cost_map_reload/status`
      : `/schedule/model_cost_map_reload/status`;
    console.log("Fetching status from URL:", url);
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
    console.log(`Model cost map reload status:`, jsonData);
    return jsonData;
  } catch (error) {
    console.error("Failed to get model cost map reload status:", error);
    throw error;
  }
};
export const modelCreateCall = async (
  accessToken: string,
  formValues: Model
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/model/new` : `/model/new`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...formValues,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);

    // Close any existing messages before showing new ones
    message.destroy();

    // Sequential success messages
    NotificationsManager.success(`Model ${formValues.model_name} created successfully`);

    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelSettingsCall = async (accessToken: String) => {
  /**
   * Get all configurable params for setting a model
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/model/settings`
      : `/model/settings`;

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
  } catch (error: any) {
    console.error("Failed to get model settings:", error);
  }
};

export const modelDeleteCall = async (
  accessToken: string,
  model_id: string
) => {
  console.log(`model_id in model delete call: ${model_id}`);
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/model/delete` : `/model/delete`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        id: model_id,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const budgetDeleteCall = async (
  accessToken: string | null,
  budget_id: string
) => {
  console.log(`budget_id in budget delete call: ${budget_id}`);

  if (accessToken == null) {
    return;
  }

  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/budget/delete`
      : `/budget/delete`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        id: budget_id,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    console.log("API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const budgetCreateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in budgetCreateCall:", formValues); // Log the form values before making the API call

    console.log("Form Values after check:", formValues);
    const url = proxyBaseUrl ? `${proxyBaseUrl}/budget/new` : `/budget/new`;
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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const budgetUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in budgetUpdateCall:", formValues); // Log the form values before making the API call

    console.log("Form Values after check:", formValues);
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/budget/update`
      : `/budget/update`;
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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const invitationCreateCall = async (
  accessToken: string,
  userID: string // Assuming formValues is an object
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/invitation/new`
      : `/invitation/new`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userID, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const invitationClaimCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in invitationCreateCall:", formValues); // Log the form values before making the API call

    console.log("Form Values after check:", formValues);
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/invitation/claim`
      : `/invitation/claim`;
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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const alertingSettingsCall = async (accessToken: String) => {
  /**
   * Get all configurable params for setting a model
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/alerting/settings`
      : `/alerting/settings`;

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

export const keyCreateServiceAccountCall = async (
  accessToken: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in keyCreateServiceAccountCall:", formValues); // Log the form values before making the API call

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
        console.log(`formValues.${field}:`, formValues[field]);
        // if there's an exception JSON.parse, show it in the message
        try {
          formValues[field] = JSON.parse(formValues[field]);
        } catch (error) {
          throw new Error(`Failed to parse ${field}: ` + error);
        }
      }
    }

    console.log("Form Values after check:", formValues);
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
    console.log("API Response:", data);
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
    console.log("Form Values in keyCreateCall:", formValues); // Log the form values before making the API call

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
        console.log(`formValues.${field}:`, formValues[field]);
        // if there's an exception JSON.parse, show it in the message
        try {
          formValues[field] = JSON.parse(formValues[field]);
        } catch (error) {
          throw new Error(`Failed to parse ${field}: ` + error);
        }
      }
    }

    console.log("Form Values after check:", formValues);
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
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const userCreateCall = async (
  accessToken: string,
  userID: string | null,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in keyCreateCall:", formValues); // Log the form values before making the API call

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
      console.log("formValues.metadata:", formValues.metadata);
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    console.log("Form Values after check:", formValues);
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
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keyDeleteCall = async (accessToken: String, user_key: String) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/key/delete` : `/key/delete`;
    console.log("in keyDeleteCall:", user_key);
    //NotificationsManager.info("Making key delete request");
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        keys: [user_key],
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log(data);
    //NotificationsManager.success("API Key Deleted");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const userDeleteCall = async (
  accessToken: string,
  userIds: string[]
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/delete` : `/user/delete`;
    console.log("in userDeleteCall:", userIds);

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_ids: userIds,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log(data);
    //NotificationsManager.success("User(s) Deleted");
    return data;
  } catch (error) {
    console.error("Failed to delete user(s):", error);
    throw error;
  }
};

export const teamDeleteCall = async (accessToken: String, teamID: String) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/team/delete` : `/team/delete`;
    console.log("in teamDeleteCall:", teamID);
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        team_ids: [teamID],
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    console.log(data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to delete key:", error);
    throw error;
  }
};

export type UserListResponse = {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  users: UserInfo[];
};

export const userListCall = async (
  accessToken: String,
  userIDs: string[] | null = null,
  page: number | null = null,
  page_size: number | null = null,
  userEmail: string | null = null,
  userRole: string | null = null,
  team: string | null = null,
  sso_user_id: string | null = null,
  sortBy: string | null = null,
  sortOrder: "asc" | "desc" | null = null
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/user/list` : `/user/list`;
    console.log("in userListCall");
    const queryParams = new URLSearchParams();

    if (userIDs && userIDs.length > 0) {
      // Convert array to comma-separated string
      const userIDsString = userIDs.join(",");
      queryParams.append("user_ids", userIDsString);
    }

    if (page) {
      queryParams.append("page", page.toString());
    }

    if (page_size) {
      queryParams.append("page_size", page_size.toString());
    }

    if (userEmail) {
      queryParams.append("user_email", userEmail);
    }

    if (userRole) {
      queryParams.append("role", userRole);
    }

    if (team) {
      queryParams.append("team", team);
    }

    if (sso_user_id) {
      queryParams.append("sso_user_ids", sso_user_id);
    }

    if (sortBy) {
      queryParams.append("sort_by", sortBy);
    }

    if (sortOrder) {
      queryParams.append("sort_order", sortOrder);
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


    const data = (await response.json()) as UserListResponse;
    console.log("/user/list API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const userInfoCall = async (
  accessToken: String,
  userID: String | null,
  userRole: String,
  viewAll: Boolean = false,
  page: number | null,
  page_size: number | null,
  lookup_user_id: boolean = false
) => {
  console.log(
    `userInfoCall: ${userID}, ${userRole}, ${viewAll}, ${page}, ${page_size}, ${lookup_user_id}`
  );
  try {
    let url: string;

    if (viewAll) {
      // Use /user/list endpoint when viewAll is true
      url = proxyBaseUrl ? `${proxyBaseUrl}/user/list` : `/user/list`;
      const queryParams = new URLSearchParams();
      if (page != null) queryParams.append("page", page.toString());
      if (page_size != null)
        queryParams.append("page_size", page_size.toString());
      url += `?${queryParams.toString()}`;
    } else {
      // Use /user/info endpoint for individual user info
      url = proxyBaseUrl ? `${proxyBaseUrl}/user/info` : `/user/info`;
      if (
        (userRole === "Admin" || userRole === "Admin Viewer") &&
        !lookup_user_id
      ) {
        // do nothing
      } else if (userID) {
        url += `?user_id=${userID}`;
      }
    }

    console.log("Requesting user data from:", url);
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
    console.log("API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch user data:", error);
    throw error;
  }
};

export const teamInfoCall = async (
  accessToken: String,
  teamID: String | null
) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/team/info` : `/team/info`;
    if (teamID) {
      url = `${url}?team_id=${teamID}`;
    }
    console.log("in teamInfoCall");
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
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
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
  accessToken: String,
  organizationID: string | null,
  userID: String | null = null,
  teamID: string | null = null,
  team_alias: string | null = null,
  page: number = 1,
  page_size: number = 10,
  sort_by: string | null = null,
  sort_order: "asc" | "desc" | null = null
): Promise<TeamListResponse> => {
  /**
   * Get list of teams with filtering and sorting options
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/v2/team/list` : `/v2/team/list`;
    console.log("in teamInfoCall");
    const queryParams = new URLSearchParams();

    if (userID) {
      queryParams.append("user_id", userID.toString());
    }

    if (organizationID) {
      queryParams.append("organization_id", organizationID.toString());
    }

    if (teamID) {
      queryParams.append("team_id", teamID.toString());
    }

    if (team_alias) {
      queryParams.append("team_alias", team_alias.toString());
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
    console.log("/v2/team/list API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const teamListCall = async (
  accessToken: String,
  organizationID: string | null,
  userID: String | null = null,
  teamID: string | null = null,
  team_alias: string | null = null
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/team/list` : `/team/list`;
    console.log("in teamInfoCall");
    const queryParams = new URLSearchParams();

    if (userID) {
      queryParams.append("user_id", userID.toString());
    }

    if (organizationID) {
      queryParams.append("organization_id", organizationID.toString());
    }

    if (teamID) {
      queryParams.append("team_id", teamID.toString());
    }

    if (team_alias) {
      queryParams.append("team_alias", team_alias.toString());
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
    console.log("/team/list API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const availableTeamListCall = async (accessToken: String) => {
  /**
   * Get all available teams on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/available`
      : `/team/available`;
    console.log("in availableTeamListCall");
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
    console.log("/team/available_teams API Response:", data);
    return data;
  } catch (error) {
    throw error;
  }
};

export const organizationListCall = async (accessToken: String) => {
  /**
   * Get all organizations on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/organization/list`
      : `/organization/list`;
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

export const organizationInfoCall = async (
  accessToken: String,
  organizationID: String
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/organization/info`
      : `/organization/info`;
    if (organizationID) {
      url = `${url}?organization_id=${organizationID}`;
    }
    console.log("in teamInfoCall");
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
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const organizationCreateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in organizationCreateCall:", formValues); // Log the form values before making the API call

    if (formValues.metadata) {
      console.log("formValues.metadata:", formValues.metadata);
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        console.error("Failed to parse metadata:", error);
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/organization/new`
      : `/organization/new`;
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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const organizationUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in organizationUpdateCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/organization/update`
      : `/organization/update`;
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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    console.log("Update Team Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const organizationDeleteCall = async (
  accessToken: string,
  organizationID: string
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/organization/delete`
      : `/organization/delete`;
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

export const transformRequestCall = async (
  accessToken: String,
  request: object
) => {
  /**
   * Transform request
   */

  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/utils/transform_request`
      : `/utils/transform_request`;

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

export const userDailyActivityCall = async (
  accessToken: String,
  startTime: Date,
  endTime: Date,
  page: number = 1
) => {
  /**
   * Get daily user activity on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/daily/activity`
      : `/user/daily/activity`;
    const queryParams = new URLSearchParams();
    queryParams.append("start_date", formatDate(startTime));
    queryParams.append("end_date", formatDate(endTime));
    queryParams.append("page_size", "1000");
    queryParams.append("page", page.toString());
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
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const tagDailyActivityCall = async (
  accessToken: String,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  tags: string[] | null = null
) => {
  /**
   * Get daily user activity on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/tag/daily/activity`
      : `/tag/daily/activity`;
    const queryParams = new URLSearchParams();
    queryParams.append("start_date", formatDate(startTime));
    queryParams.append("end_date", formatDate(endTime));
    queryParams.append("page_size", "1000");
    queryParams.append("page", page.toString());
    if (tags) {
      queryParams.append("tags", tags.join(","));
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
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const teamDailyActivityCall = async (
  accessToken: String,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  teamIds: string[] | null = null
) => {
  /**
   * Get daily user activity on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/daily/activity`
      : `/team/daily/activity`;
    const queryParams = new URLSearchParams();
    queryParams.append("start_date", formatDate(startTime));
    queryParams.append("end_date", formatDate(endTime));
    queryParams.append("page_size", "1000");
    queryParams.append("page", page.toString());
    if (teamIds) {
      queryParams.append("team_ids", teamIds.join(","));
    }
    queryParams.append("exclude_team_ids", "litellm-dashboard");
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
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const getTotalSpendCall = async (accessToken: String) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/global/spend` : `/global/spend`;

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
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const getOnboardingCredentials = async (inviteUUID: String) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/onboarding/get_token`
      : `/onboarding/get_token`;
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
  password: String
) => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/onboarding/claim_token`
    : `/onboarding/claim_token`;
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        invitation_link: inviteUUID,
        user_id: userID,
        password: password,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    console.log(data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to delete key:", error);
    throw error;
  }
};

export const regenerateKeyCall = async (
  accessToken: string,
  keyToRegenerate: string,
  formData: any
) => {
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
    console.log("Regenerate key Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to regenerate key:", error);
    throw error;
  }
};

let ModelListerrorShown = false;
let errorTimer: NodeJS.Timeout | null = null;

export const modelInfoCall = async (
  accessToken: String,
  userID: String,
  userRole: String
) => {
  /**
   * Get all models on proxy
   */
  try {
    console.log("modelInfoCall:", accessToken, userID, userRole);
    let url = proxyBaseUrl ? `${proxyBaseUrl}/v2/model/info` : `/v2/model/info`;
    const params = new URLSearchParams();
    params.append("include_team_models", "true");
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
    console.log("modelInfoCall:", data);
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelInfoV1Call = async (accessToken: String, modelId: String) => {
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
    console.log("modelInfoV1Call:", data);
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
  return response.json();
};

export const modelHubCall = async (accessToken: String) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/model_group/info`
      : `/model_group/info`;

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
    console.log("modelHubCall:", data);
    //NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

// Function to get allowed IPs
export const getAllowedIPs = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/get/allowed_ips`
      : `/get/allowed_ips`;

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
    console.log("getAllowedIPs:", data);
    return data.data; // Assuming the API returns { data: [...] }
  } catch (error) {
    console.error("Failed to get allowed IPs:", error);
    throw error;
  }
};

// Function to add an allowed IP
export const addAllowedIP = async (accessToken: String, ip: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/add/allowed_ip`
      : `/add/allowed_ip`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ip: ip }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("addAllowedIP:", data);
    return data;
  } catch (error) {
    console.error("Failed to add allowed IP:", error);
    throw error;
  }
};

// Function to delete an allowed IP
export const deleteAllowedIP = async (accessToken: String, ip: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/delete/allowed_ip`
      : `/delete/allowed_ip`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ip: ip }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("deleteAllowedIP:", data);
    return data;
  } catch (error) {
    console.error("Failed to delete allowed IP:", error);
    throw error;
  }
};

export const modelMetricsCall = async (
  accessToken: String,
  userID: String,
  userRole: String,
  modelGroup: String | null,
  startTime: String | undefined,
  endTime: String | undefined,
  apiKey: String | null,
  customer: String | null
) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/model/metrics` : `/model/metrics`;
    if (modelGroup) {
      url = `${url}?_selected_model_group=${modelGroup}&startTime=${startTime}&endTime=${endTime}&api_key=${apiKey}&customer=${customer}`;
    }
    // NotificationsManager.info("Requesting model data");
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
    // NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};
export const streamingModelMetricsCall = async (
  accessToken: String,
  modelGroup: String | null,
  startTime: String | undefined,
  endTime: String | undefined
) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/model/streaming_metrics`
      : `/model/streaming_metrics`;
    if (modelGroup) {
      url = `${url}?_selected_model_group=${modelGroup}&startTime=${startTime}&endTime=${endTime}`;
    }
    // NotificationsManager.info("Requesting model data");
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
    // NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelMetricsSlowResponsesCall = async (
  accessToken: String,
  userID: String,
  userRole: String,
  modelGroup: String | null,
  startTime: String | undefined,
  endTime: String | undefined,
  apiKey: String | null,
  customer: String | null
) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/model/metrics/slow_responses`
      : `/model/metrics/slow_responses`;
    if (modelGroup) {
      url = `${url}?_selected_model_group=${modelGroup}&startTime=${startTime}&endTime=${endTime}&api_key=${apiKey}&customer=${customer}`;
    }

    // NotificationsManager.info("Requesting model data");
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
    // NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelExceptionsCall = async (
  accessToken: String,
  userID: String,
  userRole: String,
  modelGroup: String | null,
  startTime: String | undefined,
  endTime: String | undefined,
  apiKey: String | null,
  customer: String | null
) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/model/metrics/exceptions`
      : `/model/metrics/exceptions`;

    if (modelGroup) {
      url = `${url}?_selected_model_group=${modelGroup}&startTime=${startTime}&endTime=${endTime}&api_key=${apiKey}&customer=${customer}`;
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
    // NotificationsManager.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const updateUsefulLinksCall = async (accessToken: String, useful_links: Record<string, string>) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/model_hub/update_useful_links` : `/model_hub/update_useful_links`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ useful_links: useful_links }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelAvailableCall = async (
  accessToken: String,
  userID: String,
  userRole: String,
  return_wildcard_routes: boolean = false,
  teamID: String | null = null,
  include_model_access_groups: boolean = false,
  only_model_access_groups: boolean = false
) => {
  /**
   * Get all the models user has access to
   */
  console.log(
    "in /models calls, globalLitellmHeaderName",
    globalLitellmHeaderName
  );
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/models` : `/models`;
    const params = new URLSearchParams();
    params.append("include_model_access_groups", "True");
    if (return_wildcard_routes === true) {
      params.append("return_wildcard_routes", "True");
    }
    if (only_model_access_groups === true) {
      params.append("only_model_access_groups", "True");
    }
    if (teamID) {
      params.append("team_id", teamID.toString());
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
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keySpendLogsCall = async (accessToken: String, token: String) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/logs`
      : `/global/spend/logs`;
    console.log("in keySpendLogsCall:", url);
    const response = await fetch(`${url}?api_key=${token}`, {
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const teamSpendLogsCall = async (accessToken: String) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/teams`
      : `/global/spend/teams`;
    console.log("in teamSpendLogsCall:", url);
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const tagsSpendLogsCall = async (
  accessToken: String,
  startTime: String | undefined,
  endTime: String | undefined,
  tags: String[] | undefined
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/tags`
      : `/global/spend/tags`;

    if (startTime && endTime) {
      url = `${url}?start_date=${startTime}&end_date=${endTime}`;
    }

    // if tags, convert the list to a comma separated string
    if (tags) {
      url += `${url}&tags=${tags.join(",")}`;
    }

    console.log("in tagsSpendLogsCall:", url);
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const allTagNamesCall = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/all_tag_names`
      : `/global/spend/all_tag_names`;

    console.log("in global/spend/all_tag_names call", url);
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const allEndUsersCall = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/all_end_users`
      : `/global/all_end_users`;

    console.log("in global/all_end_users call", url);
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const userFilterUICall = async (
  accessToken: String,
  params: URLSearchParams
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/filter/ui`
      : `/user/filter/ui`;

    if (params.get("user_email")) {
      url += `?user_email=${params.get("user_email")}`;
    }
    if (params.get("user_id")) {
      url += `?user_id=${params.get("user_id")}`;
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

    return await response.json();
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const userSpendLogsCall = async (
  accessToken: String,
  token: String,
  userRole: String,
  userID: String,
  startTime: String,
  endTime: String
) => {
  try {
    console.log(`user role in spend logs call: ${userRole}`);
    let url = proxyBaseUrl ? `${proxyBaseUrl}/spend/logs` : `/spend/logs`;
    if (userRole == "App Owner") {
      url = `${url}?user_id=${userID}&start_date=${startTime}&end_date=${endTime}`;
    } else {
      url = `${url}?start_date=${startTime}&end_date=${endTime}`;
    }
    //NotificationsManager.info("Making spend logs request");
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
    console.log(data);
    //NotificationsManager.success("Spend Logs received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const uiSpendLogsCall = async (
  accessToken: String,
  api_key?: string,
  team_id?: string,
  request_id?: string,
  start_date?: string,
  end_date?: string,
  page?: number,
  page_size?: number,
  user_id?: string,
  end_user?: string,
  status_filter?: string,
  model?: string,
  keyAlias?: string
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl ? `${proxyBaseUrl}/spend/logs/ui` : `/spend/logs/ui`;

    // Add query parameters if they exist
    const queryParams = new URLSearchParams();
    if (api_key) queryParams.append("api_key", api_key);
    if (team_id) queryParams.append("team_id", team_id);
    if (request_id) queryParams.append("request_id", request_id);
    if (start_date) queryParams.append("start_date", start_date);
    if (end_date) queryParams.append("end_date", end_date);
    if (page) queryParams.append("page", page.toString());
    if (page_size) queryParams.append("page_size", page_size.toString());
    if (user_id) queryParams.append("user_id", user_id);
    if (end_user) queryParams.append("end_user", end_user);
    if (status_filter) queryParams.append("status_filter", status_filter);
    if (model) queryParams.append("model", model);
    if (keyAlias) queryParams.append("key_alias", keyAlias);
    // Append query parameters to URL if any exist
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
    console.log("Spend Logs Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch spend logs:", error);
    throw error;
  }
};

export const adminSpendLogsCall = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/logs`
      : `/global/spend/logs`;

    //NotificationsManager.info("Making spend logs request");
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
    console.log(data);
    //NotificationsManager.success("Spend Logs received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const adminTopKeysCall = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/keys?limit=5`
      : `/global/spend/keys?limit=5`;

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
    console.log(data);
    //NotificationsManager.success("Spend Logs received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const adminTopEndUsersCall = async (
  accessToken: String,
  keyToken: String | null,
  startTime: String | undefined,
  endTime: String | undefined
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/end_users`
      : `/global/spend/end_users`;

    let body = "";
    if (keyToken) {
      body = JSON.stringify({
        api_key: keyToken,
        startTime: startTime,
        endTime: endTime,
      });
    } else {
      body = JSON.stringify({ startTime: startTime, endTime: endTime });
    }

    //NotificationsManager.info("Making top end users request");

    // Define requestOptions with body as an optional property
    const requestOptions = {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: body,
    };

    const response = await fetch(url, requestOptions);
    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log(data);
    //NotificationsManager.success("Top End users received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const adminspendByProvider = async (
  accessToken: String,
  keyToken: String | null,
  startTime: String | undefined,
  endTime: String | undefined
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/provider`
      : `/global/spend/provider`;

    if (startTime && endTime) {
      url += `?start_date=${startTime}&end_date=${endTime}`;
    }

    if (keyToken) {
      url += `&api_key=${keyToken}`;
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminGlobalActivity = async (
  accessToken: String,
  startTime: String | undefined,
  endTime: String | undefined
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/activity`
      : `/global/activity`;

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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminGlobalCacheActivity = async (
  accessToken: String,
  startTime: String | undefined,
  endTime: String | undefined
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/activity/cache_hits`
      : `/global/activity/cache_hits`;

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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminGlobalActivityPerModel = async (
  accessToken: String,
  startTime: String | undefined,
  endTime: String | undefined
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/activity/model`
      : `/global/activity/model`;

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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminGlobalActivityExceptions = async (
  accessToken: String,
  startTime: String | undefined,
  endTime: String | undefined,
  modelGroup: String
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/activity/exceptions`
      : `/global/activity/exceptions`;

    if (startTime && endTime) {
      url += `?start_date=${startTime}&end_date=${endTime}`;
    }

    if (modelGroup) {
      url += `&model_group=${modelGroup}`;
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminGlobalActivityExceptionsPerDeployment = async (
  accessToken: String,
  startTime: String | undefined,
  endTime: String | undefined,
  modelGroup: String
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/activity/exceptions/deployment`
      : `/global/activity/exceptions/deployment`;

    if (startTime && endTime) {
      url += `?start_date=${startTime}&end_date=${endTime}`;
    }

    if (modelGroup) {
      url += `&model_group=${modelGroup}`;
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to fetch spend data:", error);
    throw error;
  }
};

export const adminTopModelsCall = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/models?limit=5`
      : `/global/spend/models?limit=5`;

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
    console.log(data);
    //NotificationsManager.success("Top Models received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keyInfoCall = async (accessToken: String, keys: String[]) => {
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
    console.log(data);
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
  mode: string
) => {
  try {
    console.log(
      "Sending model connection test request:",
      JSON.stringify(litellm_params)
    );

    // Construct the URL based on environment
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/health/test_connection`
      : `/health/test_connection`;

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
        `Received non-JSON response (${response.status}: ${response.statusText}). Check network tab for details.`
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
          message:
            data.error?.message ||
            `Connection test failed: ${response.status} ${response.statusText}`,
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

// ... existing code ...
export const keyInfoV1Call = async (accessToken: string, key: string) => {
  try {
    console.log("entering keyInfoV1Call");
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

    console.log("response", response);

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      NotificationsManager.fromBackend("Failed to fetch key info - " + errorData);
    }

    const data = await response.json();
    console.log("data", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch key info:", error);
    throw error;
  }
};

export const keyListCall = async (
  accessToken: String,
  organizationID: string | null,
  teamID: string | null,
  selectedKeyAlias: string | null,
  userID: string | null,
  keyHash: string | null,
  page: number,
  pageSize: number,
  sortBy: string | null = null,
  sortOrder: string | null = null
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/key/list` : `/key/list`;
    console.log("in keyListCall");
    const queryParams = new URLSearchParams();

    if (teamID) {
      queryParams.append("team_id", teamID.toString());
    }

    if (organizationID) {
      queryParams.append("organization_id", organizationID.toString());
    }

    if (selectedKeyAlias) {
      queryParams.append("key_alias", selectedKeyAlias);
    }

    if (keyHash) {
      queryParams.append("key_hash", keyHash);
    }

    if (userID) {
      queryParams.append("user_id", userID.toString());
    }

    if (page) {
      queryParams.append("page", page.toString());
    }

    if (pageSize) {
      queryParams.append("size", pageSize.toString());
    }

    if (sortBy) {
      queryParams.append("sort_by", sortBy);
    }

    if (sortOrder) {
      queryParams.append("sort_order", sortOrder);
    }
    queryParams.append("return_full_object", "true");
    queryParams.append("include_team_keys", "true");
    queryParams.append("include_created_by_keys", "true");

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
    console.log("/team/list API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const spendUsersCall = async (accessToken: String, userID: String) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/spend/users` : `/spend/users`;
    console.log("in spendUsersCall:", url);
    const response = await fetch(`${url}?user_id=${userID}`, {
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
    console.log(data);
    return data;
  } catch (error) {
    console.error("Failed to get spend for user", error);
    throw error;
  }
};

export const userRequestModelCall = async (
  accessToken: String,
  model: String,
  UserID: String,
  justification: String
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/request_model`
      : `/user/request_model`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        models: [model],
        user_id: UserID,
        justification: justification,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    console.log(data);
    //NotificationsManager.success("");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const userGetRequesedtModelsCall = async (accessToken: String) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/get_requests`
      : `/user/get_requests`;
    console.log("in userGetRequesedtModelsCall:", url);
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
    console.log(data);
    //NotificationsManager.success("");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get requested models:", error);
    throw error;
  }
};

export interface User {
  user_role: string;
  user_id: string;
  user_email: string;
  [key: string]: string; // Include any other potential keys in the dictionary
}

export const userDailyActivityAggregatedCall = async (
  accessToken: String,
  startTime: Date,
  endTime: Date
) => {
  /**
   * Get aggregated daily user activity (no pagination)
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/daily/activity/aggregated`
      : `/user/daily/activity/aggregated`;
    const queryParams = new URLSearchParams();
    // Format dates as YYYY-MM-DD for the API
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    };
    queryParams.append("start_date", formatDate(startTime));
    queryParams.append("end_date", formatDate(endTime));
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
    console.error("Failed to fetch aggregated user daily activity:", error);
    throw error;
  }
};

export const userGetAllUsersCall = async (
  accessToken: String,
  role: String
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/get_users?role=${role}`
      : `/user/get_users?role=${role}`;
    console.log("in userGetAllUsersCall:", url);
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
    console.log(data);
    //NotificationsManager.success("Got all users");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get requested models:", error);
    throw error;
  }
};

export const getPossibleUserRoles = async (accessToken: String) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/available_roles`
      : `/user/available_roles`;
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

    const data = (await response.json()) as Record<
      string,
      Record<string, string>
    >;
    console.log("response from user/available_role", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    throw error;
  }
};

export const teamCreateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in teamCreateCall:", formValues); // Log the form values before making the API call
    if (formValues.metadata) {
      console.log("formValues.metadata:", formValues.metadata);
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const url = proxyBaseUrl ? `${proxyBaseUrl}/team/new` : `/team/new`;
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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const credentialCreateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in credentialCreateCall:", formValues); // Log the form values before making the API call
    if (formValues.metadata) {
      console.log("formValues.metadata:", formValues.metadata);
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const url = proxyBaseUrl ? `${proxyBaseUrl}/credentials` : `/credentials`;
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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const credentialListCall = async (accessToken: String) => {
  /**
   * Get all available teams on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/credentials` : `/credentials`;
    console.log("in credentialListCall");

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
    console.log("/credentials API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const credentialGetCall = async (
  accessToken: String,
  credentialName: String | null,
  modelId: String | null
) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/credentials` : `/credentials`;

    if (credentialName) {
      url += `/by_name/${credentialName}`;
    } else if (modelId) {
      url += `/by_model/${modelId}`;
    }

    console.log("in credentialListCall");

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
    console.log("/credentials API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const credentialDeleteCall = async (
  accessToken: String,
  credentialName: String
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/credentials/${credentialName}`
      : `/credentials/${credentialName}`;
    console.log("in credentialDeleteCall:", credentialName);
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
    console.log(data);
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
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in credentialUpdateCall:", formValues); // Log the form values before making the API call
    if (formValues.metadata) {
      console.log("formValues.metadata:", formValues.metadata);
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/credentials/${credentialName}`
      : `/credentials/${credentialName}`;
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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keyUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in keyUpdateCall:", formValues); // Log the form values before making the API call

    if (formValues.model_tpm_limit) {
      console.log("formValues.model_tpm_limit:", formValues.model_tpm_limit);
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.model_tpm_limit = JSON.parse(formValues.model_tpm_limit);
      } catch (error) {
        throw new Error("Failed to parse model_tpm_limit: " + error);
      }
    }

    if (formValues.model_rpm_limit) {
      console.log("formValues.model_rpm_limit:", formValues.model_rpm_limit);
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
    console.log("Update key Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const teamUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in teamUpateCall:", formValues); // Log the form values before making the API call

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
    console.log("Update Team Response:", data);
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
  modelId: string
) => {
  try {
    console.log("Form Values in modelUpateCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/model/${modelId}/update`
      : `/model/${modelId}/update`;
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
    console.log("Update model Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to update model:", error);
    throw error;
  }
};

export const modelUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in modelUpateCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl ? `${proxyBaseUrl}/model/update` : `/model/update`;
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
      console.error("Error update from the server:", errorData);
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    console.log("Update model Response:", data);
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
}

export const teamMemberAddCall = async (
  accessToken: string,
  teamId: string,
  formValues: Member
) => {
  try {
    console.log("Form Values in teamMemberAddCall:", formValues);

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/member_add`
      : `/team/member_add`;

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

      const rawMessage =
        parsedError?.detail?.error || "Failed to add team member";
      const err = new Error(rawMessage);
      (err as any).raw = parsedError;
      throw err;
    }

    const data = await response.json();
    console.log("API Response:", data);
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
  allUsers?: boolean
) => {
  try {
    console.log("Bulk add team members:", { teamId, members, maxBudgetInTeam });

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/bulk_member_add`
      : `/team/bulk_member_add`;

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

      const rawMessage =
        parsedError?.detail?.error || "Failed to bulk add team members";
      const err = new Error(rawMessage);
      (err as any).raw = parsedError;
      throw err;
    }

    const data = await response.json();
    console.log("Bulk team member add API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to bulk add team members:", error);
    throw error;
  }
};

export const teamMemberUpdateCall = async (
  accessToken: string,
  teamId: string,
  formValues: Member // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in teamMemberUpdateCall:", formValues);
    console.log("Budget value:", formValues.max_budget_in_team);
    console.log("TPM limit:", formValues.tpm_limit);
    console.log("RPM limit:", formValues.rpm_limit);

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/member_update`
      : `/team/member_update`;
    
    const requestBody: any = {
      team_id: teamId,
      role: formValues.role,
      user_id: formValues.user_id,
    };

    // Add optional budget and rate limit fields
    if (formValues.user_email !== undefined) {
      requestBody.user_email = formValues.user_email;
    }
    if (formValues.max_budget_in_team !== undefined && formValues.max_budget_in_team !== null) {
      requestBody.max_budget_in_team = formValues.max_budget_in_team;
    }
    if (formValues.tpm_limit !== undefined && formValues.tpm_limit !== null) {
      requestBody.tpm_limit = formValues.tpm_limit;
    }
    if (formValues.rpm_limit !== undefined && formValues.rpm_limit !== null) {
      requestBody.rpm_limit = formValues.rpm_limit;
    }

    console.log("Final request body:", requestBody);

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

      const rawMessage =
        parsedError?.detail?.error || "Failed to add team member";
      const err = new Error(rawMessage);
      (err as any).raw = parsedError;
      throw err;
    }

    const data = await response.json();
    console.log("API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to update team member:", error);
    throw error;
  }
};

export const teamMemberDeleteCall = async (
  accessToken: string,
  teamId: string,
  formValues: Member // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in teamMemberAddCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/member_delete`
      : `/team/member_delete`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        team_id: teamId,
        ...(formValues.user_email !== undefined && {
          user_email: formValues.user_email,
        }),
        ...(formValues.user_id !== undefined && {
          user_id: formValues.user_id,
        }),
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
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
  formValues: Member // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in teamMemberAddCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/organization/member_add`
      : `/organization/member_add`;
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
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create organization member:", error);
    throw error;
  }
};

export const organizationMemberDeleteCall = async (
  accessToken: string,
  organizationId: string,
  userId: string
) => {
  try {
    console.log("Form Values in organizationMemberDeleteCall:", userId); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/organization/member_delete`
      : `/organization/member_delete`;

    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        organization_id: organizationId,
        user_id: userId,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to delete organization member:", error);
    throw error;
  }
};
export const organizationMemberUpdateCall = async (
  accessToken: string,
  organizationId: string,
  formValues: Member // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in organizationMemberUpdateCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/organization/member_update`
      : `/organization/member_update`;

    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        organization_id: organizationId,
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to update organization member:", error);
    throw error;
  }
};

export const userUpdateUserCall = async (
  accessToken: string,
  formValues: any, // Assuming formValues is an object
  userRole: string | null
) => {
  try {
    console.log("Form Values in userUpdateUserCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/update` : `/user/update`;
    let response_body = { ...formValues };
    if (userRole !== null) {
      response_body["user_role"] = userRole;
    }
    response_body = JSON.stringify(response_body);
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: response_body,
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = (await response.json()) as {
      user_id: string;
      data: UserInfo;
    };
    console.log("API Response:", data);
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
  allUsers: boolean = false // Flag to update all users
) => {
  try {
    console.log("Form Values in userUpdateUserCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/bulk_update`
      : `/user/bulk_update`;
    
    let request_body_json: string;
    
    if (allUsers) {
      // Update all users mode
      request_body_json = JSON.stringify({
        all_users: true,
        user_updates: formValues,
      });
    } else if (userIds && userIds.length > 0) {
      // Update specific users mode
      let request_body = [] 
      for (const user_id of userIds) {
        request_body.push({
          user_id: user_id,
          ...formValues,
        });
      }
      request_body_json = JSON.stringify({
        users: request_body,
      });
    } else {
      throw new Error("Must provide either userIds or set allUsers=true");
    }
    
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: request_body_json,
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = (await response.json()) as {
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
    console.log("API Response:", data);
    //NotificationsManager.success("User role updated");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const PredictedSpendLogsCall = async (
  accessToken: string,
  requestData: any
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/predict/spend/logs`
      : `/global/predict/spend/logs`;

    //NotificationsManager.info("Predicting spend logs request");

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data: requestData,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log(data);
    //NotificationsManager.success("Predicted Logs received");
    return data;
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const slackBudgetAlertsHealthCheck = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/health/services?service=slack_budget_alerts`
      : `/health/services?service=slack_budget_alerts`;

    console.log("Checking Slack Budget Alerts service health");
    //NotificationsManager.info("Sending Test Slack alert...");

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
    NotificationsManager.success("Test Slack Alert worked - check your Slack!");
    console.log("Service Health Response:", data);

    // You can add additional logic here based on the response if needed

    return data;
  } catch (error) {
    console.error("Failed to perform health check:", error);
    throw error;
  }
};

export const serviceHealthCheck = async (
  accessToken: String,
  service: String
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/health/services?service=${service}`
      : `/health/services?service=${service}`;

    console.log("Checking Slack Budget Alerts service health");

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

export const getBudgetList = async (accessToken: String) => {
  /**
   * Get all configurable params for setting a budget
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/budget/list` : `/budget/list`;

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
export const getBudgetSettings = async (accessToken: String) => {
  /**
   * Get all configurable params for setting a budget
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/budget/settings`
      : `/budget/settings`;

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

export const getCallbacksCall = async (
  accessToken: String,
  userID: String,
  userRole: String
) => {
  /**
   * Get all the models user has access to
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/get/config/callbacks`
      : `/get/config/callbacks`;

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

export const getGeneralSettingsCall = async (accessToken: String) => {
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

export const getPassThroughEndpointsCall = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/pass_through_endpoint`
      : `/config/pass_through_endpoint`;

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

export const getConfigFieldSetting = async (
  accessToken: String,
  fieldName: string
) => {
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

export const updatePassThroughFieldSetting = async (
  accessToken: String,
  fieldName: string,
  fieldValue: any
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/pass_through_endpoint`
      : `/config/pass_through_endpoint`;

    let formData = {
      field_name: fieldName,
      field_value: fieldValue,
    };
    //NotificationsManager.info("Requesting model data");
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
    //NotificationsManager.info("Received model data");
    NotificationsManager.success("Successfully updated value!");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to set callbacks:", error);
    throw error;
  }
};

export const createPassThroughEndpoint = async (
  accessToken: String,
  formValues: Record<string, any>
) => {
  /**
   * Set callbacks on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/pass_through_endpoint`
      : `/config/pass_through_endpoint`;

    //NotificationsManager.info("Requesting model data");
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
    console.error("Failed to set callbacks:", error);
    throw error;
  }
};

export const updateConfigFieldSetting = async (
  accessToken: String,
  fieldName: string,
  fieldValue: any
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/field/update`
      : `/config/field/update`;

    let formData = {
      field_name: fieldName,
      field_value: fieldValue,
      config_type: "general_settings",
    };
    //NotificationsManager.info("Requesting model data");
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
    //NotificationsManager.info("Received model data");
    NotificationsManager.success("Successfully updated value!");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to set callbacks:", error);
    throw error;
  }
};

export const deleteConfigFieldSetting = async (
  accessToken: String,
  fieldName: String
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/field/delete`
      : `/config/field/delete`;

    let formData = {
      field_name: fieldName,
      config_type: "general_settings",
    };
    //NotificationsManager.info("Requesting model data");
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
    NotificationsManager.success("Field reset on proxy");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const deletePassThroughEndpointsCall = async (
  accessToken: String,
  endpointId: string
) => {
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

export const setCallbacksCall = async (
  accessToken: String,
  formValues: Record<string, any>
) => {
  /**
   * Set callbacks on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/config/update` : `/config/update`;

    //NotificationsManager.info("Requesting model data");
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
    console.error("Failed to set callbacks:", error);
    throw error;
  }
};

export const healthCheckCall = async (accessToken: String) => {
  /**
   * Get all the models user has access to
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/health` : `/health`;

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
    console.error("Failed to call /health:", error);
    throw error;
  }
};

export const individualModelHealthCheckCall = async (
  accessToken: String,
  modelName: string
) => {
  /**
   * Run health check for a specific model using model name
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/health?model=${encodeURIComponent(modelName)}`
      : `/health?model=${encodeURIComponent(modelName)}`;

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
    console.error(`Failed to call /health for model ${modelName}:`, error);
    throw error;
  }
};

export const cachingHealthCheckCall = async (accessToken: String) => {
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

export const healthCheckHistoryCall = async (
  accessToken: String,
  model?: string,
  statusFilter?: string,
  limit: number = 100,
  offset: number = 0
) => {
  /**
   * Get health check history for models
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/health/history`
      : `/health/history`;

    const params = new URLSearchParams();
    if (model) params.append("model", model);
    if (statusFilter) params.append("status_filter", statusFilter);
    params.append("limit", limit.toString());
    params.append("offset", offset.toString());

    if (params.toString()) {
      url += `?${params.toString()}`;
    }

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
    console.error("Failed to call /health/history:", error);
    throw error;
  }
};


export const latestHealthChecksCall = async (accessToken: String) => {
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

export const getProxyUISettings = async (accessToken: String) => {
  /**
   * Get all the models user has access to
   */
  try {
    console.log("Getting proxy UI settings");
    console.log("proxyBaseUrl in getProxyUISettings:", proxyBaseUrl);
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/sso/get/ui_settings`
      : `/sso/get/ui_settings`;

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

export const getGuardrailsList = async (accessToken: String) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/v2/guardrails/list`
      : `/v2/guardrails/list`;
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
    console.error("Failed to get guardrails list:", error);
    throw error;
  }
};

export const getPromptsList = async (accessToken: String) : Promise<ListPromptsResponse> => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/prompts/list`
      : `/prompts/list`;
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
    console.error("Failed to get prompts list:", error);
    throw error;
  }
};

export const getPromptInfo = async (accessToken: String, promptId: string): Promise<PromptInfoResponse> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/prompts/${promptId}/info` : `/prompts/${promptId}/info`;
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
    console.error("Failed to get prompt info:", error);
    throw error;
  }
};

export const createPromptCall = async (
  accessToken: string,
  promptData: any
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/prompts` : `/prompts`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(promptData),
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
    console.error("Failed to create prompt:", error);
    throw error;
  }
};

export const updatePromptCall = async (
  accessToken: string,
  promptId: string,
  promptData: any
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/prompts/${promptId}` : `/prompts/${promptId}`;

    const response = await fetch(url, {
      method: "PUT",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(promptData),
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
    console.error("Failed to update prompt:", error);
    throw error;
  }
};

export const deletePromptCall = async (
  accessToken: string,
  promptId: string
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/prompts/${promptId}` : `/prompts/${promptId}`;

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
    console.error("Failed to delete prompt:", error);
    throw error;
  }
};

export const convertPromptFileToJson = async (
  accessToken: string,
  file: File
): Promise<{ prompt_id: string; json_data: any }> => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    
    const url = proxyBaseUrl 
      ? `${proxyBaseUrl}/utils/dotprompt_json_converter` 
      : `/utils/dotprompt_json_converter`;
    
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

export const patchPromptCall = async (
  accessToken: string,
  promptId: string,
  promptData: any
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/prompts/${promptId}` : `/prompts/${promptId}`;

    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(promptData),
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
    console.error("Failed to patch prompt:", error);
    throw error;
  }
};

export const createGuardrailCall = async (
  accessToken: string,
  guardrailData: any
) => {
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
    console.log("Create guardrail response:", data);
    return data;
  } catch (error) {
    console.error("Failed to create guardrail:", error);
    throw error;
  }
};

export const uiSpendLogDetailsCall = async (
  accessToken: string,
  logId: string,
  start_date: string
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/spend/logs/ui/${logId}?start_date=${encodeURIComponent(start_date)}`
      : `/spend/logs/ui/${logId}?start_date=${encodeURIComponent(start_date)}`;

    console.log("Fetching log details from:", url);

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
    console.log("Fetched log details:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch log details:", error);
    throw error;
  }
};

export const getInternalUserSettings = async (accessToken: string) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/get/internal_user_settings`
      : `/get/internal_user_settings`;

    console.log("Fetching SSO settings from:", url);

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
    console.log("Fetched SSO settings:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch SSO settings:", error);
    throw error;
  }
};

export const updateInternalUserSettings = async (
  accessToken: string,
  settings: Record<string, any>
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/update/internal_user_settings`
      : `/update/internal_user_settings`;

    console.log("Updating internal user settings:", settings);

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
    console.log("Updated internal user settings:", data);
    NotificationsManager.success("Internal user settings updated successfully");
    return data;
  } catch (error) {
    console.error("Failed to update internal user settings:", error);
    throw error;
  }
};

export const fetchMCPServers = async (accessToken: string) => {
  try {
    // Construct base URL
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/v1/mcp/server`
      : `/v1/mcp/server`;

    console.log("Fetching MCP servers from:", url);

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
    console.log("Fetched MCP servers:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch MCP servers:", error);
    throw error;
  }
};

export const fetchMCPAccessGroups = async (accessToken: string) => {
  try {
    // Construct base URL
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/v1/mcp/access_groups`
      : `/v1/mcp/access_groups`;

    console.log("Fetching MCP access groups from:", url);

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
    console.log("Fetched MCP access groups:", data);
    return data.access_groups || [];
  } catch (error) {
    console.error("Failed to fetch MCP access groups:", error);
    throw error;
  }
};

export const createMCPServer = async (
  accessToken: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in createMCPServer:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/v1/mcp/server`
      : `/v1/mcp/server`;

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
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }

    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const updateMCPServer = async (
  accessToken: string,
  formValues: Record<string, any>
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/v1/mcp/server`
      : `/v1/mcp/server`;
    const response = await fetch(url, {
      method: "PUT",
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

    return await response.json();
  } catch (error) {
    console.error("Failed to update MCP server:", error);
    throw error;
  }
};

export const deleteMCPServer = async (
  accessToken: String,
  serverId: String
) => {
  try {
    const url =
      (proxyBaseUrl ? `${proxyBaseUrl}` : "") + `/v1/mcp/server/${serverId}`;
    console.log("in deleteMCPServer:", serverId);
    const response = await fetch(url, {
      method: HTTP_REQUEST.DELETE,
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

  } catch (error) {
    console.error("Failed to delete key:", error);
    throw error;
  }
};

export const listMCPTools = async (accessToken: string, serverId: string, authValue?: string, serverAlias?: string) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/mcp-rest/tools/list?server_id=${serverId}`
      : `/mcp-rest/tools/list?server_id=${serverId}`;

    console.log("Fetching MCP tools from:", url);

    const headers: Record<string, string> = {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    };

    // Use new server-specific auth header format if serverAlias is provided
    if (serverAlias && authValue) {
      headers[`x-mcp-${serverAlias}-authorization`] = authValue;
    } else if (authValue) {
      // Fall back to deprecated x-mcp-auth header for backward compatibility
      headers[MCP_AUTH_HEADER] = authValue;
    }

    const response = await fetch(url, {
      method: "GET",
      headers,
    });

    const data = await response.json();
    console.log("Fetched MCP tools response:", data);

    if (!response.ok) {
      // If the server returned an error response, use it
      if (data.error && data.message) {
        throw new Error(data.message);
      }
      // Otherwise use a generic error
      throw new Error("Failed to fetch MCP tools");
    }

    // Return the full response object which includes tools, error, and message
    return data;
  } catch (error) {
    console.error("Failed to fetch MCP tools:", error);
    // Return an error response in the same format as the API
    return {
      tools: [],
      error: "network_error",
      message: error instanceof Error ? error.message : "Failed to fetch MCP tools"
    };
  }
};


export const callMCPTool = async (
  accessToken: string,
  toolName: string,
  toolArguments: Record<string, any>,
  authValue: string,
  serverAlias?: string,
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/mcp-rest/tools/call`
      : `/mcp-rest/tools/call`;

    console.log(
      "Calling MCP tool:",
      toolName,
      "with arguments:",
      toolArguments
    );

    const headers: Record<string, string> = {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    };

    // Use new server-specific auth header format if serverAlias is provided
    if (serverAlias) {
      headers[`x-mcp-${serverAlias}-authorization`] = authValue;
    } else {
      // Fall back to deprecated x-mcp-auth header for backward compatibility
      headers[MCP_AUTH_HEADER] = authValue;
    }


    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({
        name: toolName,
        arguments: toolArguments,
      }),
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
          if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          } else if (typeof errorData.detail === 'object') {
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
    console.log("MCP tool call response:", data);
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

export const tagCreateCall = async (
  accessToken: string,
  formValues: TagNewRequest
): Promise<void> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/new` : `/tag/new`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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

export const tagUpdateCall = async (
  accessToken: string,
  formValues: TagUpdateRequest
): Promise<void> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/update` : `/tag/update`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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

export const tagInfoCall = async (
  accessToken: string,
  tagNames: string[]
): Promise<TagInfoResponse> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/info` : `/tag/info`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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

export const tagListCall = async (
  accessToken: string
): Promise<TagListResponse> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/list` : `/tag/list`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
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

export const tagDeleteCall = async (
  accessToken: string,
  tagName: string
): Promise<void> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/tag/delete` : `/tag/delete`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/get/default_team_settings`
      : `/get/default_team_settings`;

    console.log("Fetching default team settings from:", url);

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
    console.log("Fetched default team settings:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch default team settings:", error);
    throw error;
  }
};

export const updateDefaultTeamSettings = async (
  accessToken: string,
  settings: Record<string, any>
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/update/default_team_settings`
      : `/update/default_team_settings`;

    console.log("Updating default team settings:", settings);

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
    console.log("Updated default team settings:", data);
    NotificationsManager.success("Default team settings updated successfully");
    return data;
  } catch (error) {
    console.error("Failed to update default team settings:", error);
    throw error;
  }
};

export const getTeamPermissionsCall = async (
  accessToken: string,
  teamId: string
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/permissions_list?team_id=${teamId}`
      : `/team/permissions_list?team_id=${teamId}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("Team permissions response:", data);
    return data;
  } catch (error) {
    console.error("Failed to get team permissions:", error);
    throw error;
  }
};

export const teamPermissionsUpdateCall = async (
  accessToken: string,
  teamId: string,
  permissions: string[]
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/permissions_update`
      : `/team/permissions_update`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({
        team_id: teamId,
        team_member_permissions: permissions,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      handleError(errorMessage);
      throw new Error(errorMessage);
    }


    const data = await response.json();
    console.log("Team permissions response:", data);
    return data;
  } catch (error) {
    console.error("Failed to update team permissions:", error);
    throw error;
  }
};

/**
 * Get all spend logs for a particular session
 */
export const sessionSpendLogsCall = async (
  accessToken: string,
  session_id: string
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/spend/logs/session/ui?session_id=${encodeURIComponent(session_id)}`
      : `/spend/logs/session/ui?session_id=${encodeURIComponent(session_id)}`;

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

export const vectorStoreCreateCall = async (
  accessToken: string,
  formValues: Record<string, any>
): Promise<void> => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/vector_store/new`
      : `/vector_store/new`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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
  page_size: number = 100
): Promise<any> => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/vector_store/list`
      : `/vector_store/list`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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

export const vectorStoreDeleteCall = async (
  accessToken: string,
  vectorStoreId: string
): Promise<void> => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/vector_store/delete`
      : `/vector_store/delete`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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

export const vectorStoreInfoCall = async (
  accessToken: string,
  vectorStoreId: string
): Promise<any> => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/vector_store/info`
      : `/vector_store/info`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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

export const vectorStoreUpdateCall = async (
  accessToken: string,
  formValues: Record<string, any>
): Promise<any> => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/vector_store/update`
      : `/vector_store/update`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
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

export const getEmailEventSettings = async (
  accessToken: string
): Promise<EmailEventSettingsResponse> => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/email/event_settings`
      : `/email/event_settings`;

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
    console.log("Email event settings response:", data);
    return data;
  } catch (error) {
    console.error("Failed to get email event settings:", error);
    throw error;
  }
};

export const updateEmailEventSettings = async (
  accessToken: string,
  settings: EmailEventSettingsUpdateRequest
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/email/event_settings`
      : `/email/event_settings`;

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
    console.log("Update email event settings response:", data);
    return data;
  } catch (error) {
    console.error("Failed to update email event settings:", error);
    throw error;
  }
};

export const resetEmailEventSettings = async (accessToken: string) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/email/event_settings/reset`
      : `/email/event_settings/reset`;

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
    console.log("Reset email event settings response:", data);
    return data;
  } catch (error) {
    console.error("Failed to reset email event settings:", error);
    throw error;
  }
};

export { type UserInfo } from "./view_users/types"; // Re-export UserInfo
export { type Team } from "./key_team_helpers/key_list"; // Re-export Team

export const deleteGuardrailCall = async (
  accessToken: string,
  guardrailId: string
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/guardrails/${guardrailId}`
      : `/guardrails/${guardrailId}`;

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
    console.log("Delete guardrail response:", data);
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
    console.log("Guardrail UI settings response:", data);
    return data;
  } catch (error) {
    console.error("Failed to get guardrail UI settings:", error);
    throw error;
  }
};

export const getGuardrailProviderSpecificParams = async (
  accessToken: string
) => {
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
    console.log("Guardrail provider specific params response:", data);
    return data;
  } catch (error) {
    console.error(
      "Failed to get guardrail provider specific parameters:",
      error
    );
    throw error;
  }
};

export const getGuardrailInfo = async (
  accessToken: string,
  guardrailId: string
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/guardrails/${guardrailId}/info`
      : `/guardrails/${guardrailId}/info`;

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
    console.log("Guardrail info response:", data);
    return data;
  } catch (error) {
    console.error("Failed to get guardrail info:", error);
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
  }
) => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/guardrails/${guardrailId}`
      : `/guardrails/${guardrailId}`;

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
    console.log("Update guardrail response:", data);
    return data;
  } catch (error) {
    console.error("Failed to update guardrail:", error);
    throw error;
  }
};

export const getSSOSettings = async (accessToken: string) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/get/sso_settings`
      : `/get/sso_settings`;

    console.log("Fetching SSO configuration from:", url);

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
    console.log("Fetched SSO configuration:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch SSO configuration:", error);
    throw error;
  }
};

export const updateSSOSettings = async (
  accessToken: string,
  settings: Record<string, any>
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/update/sso_settings`
      : `/update/sso_settings`;

    console.log("Updating SSO configuration:", settings);

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
    console.log("Updated SSO configuration:", data);
    return data;
  } catch (error) {
    console.error("Failed to update SSO configuration:", error);
    throw error;
  }
};

export const uiAuditLogsCall = async (
  accessToken: String,
  start_date?: string,
  end_date?: string,
  page?: number,
  page_size?: number
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl ? `${proxyBaseUrl}/audit` : `/audit`;

    // Add query parameters if they exist
    const queryParams = new URLSearchParams();
    // if (start_date) queryParams.append('start_date', start_date);
    // if (end_date) queryParams.append('end_date', end_date);
    if (page) queryParams.append("page", page.toString());
    if (page_size) queryParams.append("page_size", page_size.toString());

    // Append query parameters to URL if any exist
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
    console.error("Failed to fetch audit logs:", error);
    throw error;
  }
};

export const getRemainingUsers = async (
  accessToken: string
): Promise<{
  total_users: number | null;
  total_users_used: number;
  total_users_remaining: number | null;
  total_teams: number | null;
  total_teams_used: number;
  total_teams_remaining: number | null;
} | null> => {
  try {
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/user/available_users`
      : `/user/available_users`;

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

export const updatePassThroughEndpoint = async (
  accessToken: string,
  endpointPath: string,
  formValues: Record<string, any>
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

export const getPassThroughEndpointInfo = async (
  accessToken: string,
  endpointPath: string
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/pass_through_endpoint?endpoint_id=${encodeURIComponent(endpointPath)}`
      : `/config/pass_through_endpoint?endpoint_id=${encodeURIComponent(endpointPath)}`;

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
    const endpoints = data["endpoints"];

    if (!endpoints || endpoints.length === 0) {
      throw new Error("Pass through endpoint not found");
    }

    return endpoints[0]; // Return the first (and should be only) endpoint
  } catch (error) {
    console.error("Failed to get pass through endpoint info:", error);
    throw error;
  }
};

export const deleteCallback = async (
  accessToken: String,
  callbackName: string
) => {
  /**
   * Delete specific callback from proxy using the /config/callback/delete API
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/callback/delete`
      : `/config/callback/delete`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        callback_name: callbackName,
      }),
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
    console.error("Failed to delete specific callback:", error);
    throw error;
  }
};

export const mcpToolsCall = async (accessToken: string) => {
  const proxyBaseUrl = getProxyBaseUrl();
  const response = await fetch(`${proxyBaseUrl}/v1/mcp/tools`, {
    method: "GET",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });



  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

export const testMCPConnectionRequest = async (
  accessToken: string,
  mcpServerConfig: Record<string, any>
) => {
  try {
    console.log(
      "Testing MCP connection with config:",
      JSON.stringify(mcpServerConfig)
    );

    // Construct the URL for POST request
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/mcp-rest/test/connection`
      : `/mcp-rest/test/connection`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(mcpServerConfig),
    });

    // Check for non-JSON responses first
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const text = await response.text();
      console.error("Received non-JSON response:", text);
      throw new Error(
        `Received non-JSON response (${response.status}: ${response.statusText}). Check network tab for details.`
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
          message:
            data.error?.message ||
            `MCP connection test failed: ${response.status} ${response.statusText}`,
        };
      }
    }

    return data;
  } catch (error) {
    console.error("MCP connection test error:", error);
    // For network errors or other exceptions, still throw
    throw error;
  }
};

export const testMCPToolsListRequest = async (
  accessToken: string,
  mcpServerConfig: Record<string, any>
) => {
  try {
    console.log(
      "Testing MCP tools list with config:",
      JSON.stringify(mcpServerConfig)
    );

    // Construct the URL for POST request
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/mcp-rest/test/tools/list`
      : `/mcp-rest/test/tools/list`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(mcpServerConfig),
    });

    // Check for non-JSON responses first
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const text = await response.text();
      console.error("Received non-JSON response:", text);
      throw new Error(
        `Received non-JSON response (${response.status}: ${response.statusText}). Check network tab for details.`
      );
    }

    const data = await response.json();

    if (!response.ok || data.error) {
      // Return the error response instead of throwing an error
      // This allows the caller to handle the error format properly
      if (data.error) {
        return data; // Return the full error response
      } else {
        return {
          tools: [],
          error: "request_failed",
          message:
            data.message ||
            `MCP tools list failed: ${response.status} ${response.statusText}`,
        };
      }
    }

    return data;
  } catch (error) {
    console.error("MCP tools list test error:", error);
    // For network errors or other exceptions, still throw
    throw error;
  }
};

export const vectorStoreSearchCall = async (
  accessToken: string,
  vectorStoreId: string,
  query: string
): Promise<any> => {
  try {
    const url = `${getProxyBaseUrl()}/v1/vector_stores/${vectorStoreId}/search`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query
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

export const userAgentAnalyticsCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  page: number = 1,
  pageSize: number = 50,
  userAgentFilter?: string
) => {
  /**
   * Get user agent analytics data including DAU, WAU, MAU, successful requests, and completed tokens
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/tag/user-agent/analytics`
      : `/tag/user-agent/analytics`;
    
    const queryParams = new URLSearchParams();
    
    // Format dates as YYYY-MM-DD for the API
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    };
    
    queryParams.append("start_date", formatDate(startTime));
    queryParams.append("end_date", formatDate(endTime));
    queryParams.append("page", page.toString());
    queryParams.append("page_size", pageSize.toString());
    
    if (userAgentFilter) {
      queryParams.append("user_agent_filter", userAgentFilter);
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
    console.error("Failed to fetch user agent analytics:", error);
    throw error;
  }
};

// New endpoint functions for DAU, WAU, MAU
export const tagDauCall = async (
  accessToken: string,
  endDate: Date,
  tagFilter?: string,
  tagFilters?: string[]
) => {
  /**
   * Get Daily Active Users (DAU) for last 7 days ending on endDate
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/tag/dau`
      : `/tag/dau`;
    
    const queryParams = new URLSearchParams();
    
    // Format date as YYYY-MM-DD for the API
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    };
    
    queryParams.append("end_date", formatDate(endDate));
    
    // Handle multiple tag filters (takes precedence over single tag filter)
    if (tagFilters && tagFilters.length > 0) {
      tagFilters.forEach(tag => {
        queryParams.append("tag_filters", tag);
      });
    } else if (tagFilter) {
      queryParams.append("tag_filter", tagFilter);
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
    console.error("Failed to fetch DAU:", error);
    throw error;
  }
};

export const tagWauCall = async (
  accessToken: string,
  endDate: Date,
  tagFilter?: string,
  tagFilters?: string[]
) => {
  /**
   * Get Weekly Active Users (WAU) for last 7 weeks ending on endDate
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/tag/wau`
      : `/tag/wau`;
    
    const queryParams = new URLSearchParams();
    
    // Format date as YYYY-MM-DD for the API
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    };
    
    queryParams.append("end_date", formatDate(endDate));
    
    // Handle multiple tag filters (takes precedence over single tag filter)
    if (tagFilters && tagFilters.length > 0) {
      tagFilters.forEach(tag => {
        queryParams.append("tag_filters", tag);
      });
    } else if (tagFilter) {
      queryParams.append("tag_filter", tagFilter);
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
    console.error("Failed to fetch WAU:", error);
    throw error;
  }
};

export const tagMauCall = async (
  accessToken: string,
  endDate: Date,
  tagFilter?: string,
  tagFilters?: string[]
) => {
  /**
   * Get Monthly Active Users (MAU) for last 7 months ending on endDate
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/tag/mau`
      : `/tag/mau`;
    
    const queryParams = new URLSearchParams();
    
    // Format date as YYYY-MM-DD for the API
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    };
    
    queryParams.append("end_date", formatDate(endDate));
    
    // Handle multiple tag filters (takes precedence over single tag filter)
    if (tagFilters && tagFilters.length > 0) {
      tagFilters.forEach(tag => {
        queryParams.append("tag_filters", tag);
      });
    } else if (tagFilter) {
      queryParams.append("tag_filter", tagFilter);
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
    console.error("Failed to fetch MAU:", error);
    throw error;
  }
};

export const tagDistinctCall = async (
  accessToken: string
) => {
  /**
   * Get all distinct user agent tags (up to 250)
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/tag/distinct`
      : `/tag/distinct`;

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
    console.error("Failed to fetch distinct tags:", error);
    throw error;
  }
};

export const userAgentSummaryCall = async (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  tagFilters?: string[]
) => {
  /**
   * Get user agent summary statistics
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/tag/summary`
      : `/tag/summary`;
    
    const queryParams = new URLSearchParams();
    
    // Format dates as YYYY-MM-DD for the API
    const formatDate = (date: Date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    };
    
    queryParams.append("start_date", formatDate(startTime));
    queryParams.append("end_date", formatDate(endTime));
    
    // Handle multiple tag filters
    if (tagFilters && tagFilters.length > 0) {
      tagFilters.forEach(tag => {
        queryParams.append("tag_filters", tag);
      });
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
    console.error("Failed to fetch user agent summary:", error);
    throw error;
  }
};

export const perUserAnalyticsCall = async (
  accessToken: string,
  page: number = 1,
  pageSize: number = 50,
  tagFilters?: string[]
) => {
  /**
   * Get per-user analytics data for the last 30 days
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/tag/user-agent/per-user-analytics`
      : `/tag/user-agent/per-user-analytics`;
    
    const queryParams = new URLSearchParams();
    
    queryParams.append("page", page.toString());
    queryParams.append("page_size", pageSize.toString());
    
    // Handle multiple tag filters
    if (tagFilters && tagFilters.length > 0) {
      tagFilters.forEach(tag => {
        queryParams.append("tag_filters", tag);
      });
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
    console.error("Failed to fetch per-user analytics:", error);
    throw error;
  }
};

const deriveErrorMessage = (errorData: any): string => {
  return (errorData?.error && (errorData.error.message || errorData.error)) ||
    errorData?.message ||
    errorData?.detail ||
    errorData?.error ||
    JSON.stringify(errorData);
};
