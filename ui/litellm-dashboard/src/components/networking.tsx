/**
 * Helper file for calls being made to proxy
 */
import { message } from "antd";

const isLocal = process.env.NODE_ENV === "development";
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;
if (isLocal != true) {
  console.log = function() {};
}

export interface Model {
  model_name: string;
  litellm_params: Object;
  model_info: Object | null;
}

const baseUrl = "/"; // Assuming the base URL is the root


let lastErrorTime = 0;

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const handleError = async (errorData: string) => {
  const currentTime = Date.now();
  if (currentTime - lastErrorTime > 60000) { // 60000 milliseconds = 60 seconds
    if (errorData.includes("Authentication Error - Expired Key")) {
      message.info("UI Session Expired. Logging out.");
      lastErrorTime = currentTime;
      await sleep(3000); // 5 second sleep
      document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
      window.location.href = baseUrl;
    }
    lastErrorTime = currentTime;
  } else {
    console.log("Error suppressed to prevent spam:", errorData);
  }
};


// Global variable for the header name
let globalLitellmHeaderName: string  = "Authorization";

// Function to set the global header name
export function setGlobalLitellmHeaderName(headerName: string = "Authorization") {
  console.log(`setGlobalLitellmHeaderName: ${headerName}`);
  globalLitellmHeaderName = headerName;
}


export const modelCostMap = async (
  accessToken: string,
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/get/litellm_model_cost_map` : `/get/litellm_model_cost_map`;
    const response = await fetch(
      url, {
        method: "GET",
        headers: {
          [globalLitellmHeaderName]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      }
    );
    const jsonData = await response.json();
    console.log(`received litellm model cost data: ${jsonData}`);
    return jsonData;
  } catch (error) {
    console.error("Failed to get model cost map:", error);
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
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log("API Response:", data);
    message.success(
      "Model created successfully"
    );
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
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
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
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
    const url = proxyBaseUrl ? `${proxyBaseUrl}/budget/update` : `/budget/update`;
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
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const keyCreateCall = async (
  accessToken: string,
  userID: string,
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
      throw new Error("Network response was not ok");
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
    //message.info("Making key delete request");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    //message.success("API Key Deleted");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const userDeleteCall = async (accessToken: string, userIds: string[]) => {
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    //message.success("User(s) Deleted");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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

export const userInfoCall = async (
  accessToken: String,
  userID: String | null,
  userRole: String,
  viewAll: Boolean = false,
  page: number | null,
  page_size: number | null
) => {
  try {
    let url: string;
    
    if (viewAll) {
      // Use /user/list endpoint when viewAll is true
      url = proxyBaseUrl ? `${proxyBaseUrl}/user/list` : `/user/list`;
      const queryParams = new URLSearchParams();
      if (page != null) queryParams.append('page', page.toString());
      if (page_size != null) queryParams.append('page_size', page_size.toString());
      url += `?${queryParams.toString()}`;
    } else {
      // Use /user/info endpoint for individual user info
      url = proxyBaseUrl ? `${proxyBaseUrl}/user/info` : `/user/info`;
      if (userRole === "Admin" || userRole === "Admin Viewer") {
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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

export const teamListCall = async (
  accessToken: String, 
  userID: String | null = null
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/team/list` : `/team/list`;
    console.log("in teamInfoCall");
    if (userID) {
      url += `?user_id=${userID}`;
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
      throw new Error("Network response was not ok");
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


export const availableTeamListCall = async (
  accessToken: String, 
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/team/available` : `/team/available`;
    console.log("in availableTeamListCall");
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
      throw new Error("Network response was not ok");
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
    let url = proxyBaseUrl ? `${proxyBaseUrl}/organization/list` : `/organization/list`;
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    return data;
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

    const url = proxyBaseUrl ? `${proxyBaseUrl}/organization/new` : `/organization/new`;
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
      throw new Error("Network response was not ok");
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

export const getTotalSpendCall = async (accessToken: String) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/global/spend` : `/global/spend`;

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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
    let url = proxyBaseUrl ? `${proxyBaseUrl}/v2/model/info` : `/v2/model/info`;

    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      let errorData = await response.text();
      errorData += `error shown=${ModelListerrorShown}`
      if (!ModelListerrorShown) {
        if (errorData.includes("No model list passed")) {
          errorData = "No Models Exist. Click Add Model to get started.";
        }
        message.info(errorData, 10);
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
    //message.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelHubCall = async (accessToken: String) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/model_group/info`
      : `/model_group/info`;

    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log("modelHubCall:", data);
    //message.info("Received model data");
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
      const errorData = await response.text();
      throw new Error(`Network response was not ok: ${errorData}`);
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
      const errorData = await response.text();
      throw new Error(`Network response was not ok: ${errorData}`);
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
      const errorData = await response.text();
      throw new Error(`Network response was not ok: ${errorData}`);
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
    // message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    // message.info("Received model data");
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
    // message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    // message.info("Received model data");
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

    // message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    // message.info("Received model data");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    // message.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const modelAvailableCall = async (
  accessToken: String,
  userID: String,
  userRole: String,
  return_wildcard_routes: boolean = false
) => {
  /**
   * Get all the models user has access to
   */
  console.log("in /models calls, globalLitellmHeaderName", globalLitellmHeaderName)
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/models` : `/models`;
    if (return_wildcard_routes === true) {
      url += `?return_wildcard_routes=True`;
    }

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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
      const errorData = await response.text();
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
      const errorData = await response.text();
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
      const errorData = await response.text();
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
      const errorData = await response.text();
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
      const errorData = await response.text();
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

export const userFilterUICall = async (accessToken: String, params: URLSearchParams) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/user/filter/ui` : `/user/filter/ui`;

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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }
  return await response.json();
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
}

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
    //message.info("Making spend logs request");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    //message.success("Spend Logs received");
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
  min_spend?: number,
  max_spend?: number,
) => {
  try {
    // Construct base URL
    let url = proxyBaseUrl ? `${proxyBaseUrl}/spend/logs/ui` : `/spend/logs/ui`;

    // Add query parameters if they exist
    const queryParams = new URLSearchParams();
    if (api_key) queryParams.append('api_key', api_key);
    if (team_id) queryParams.append('team_id', team_id);
    if (min_spend) queryParams.append('min_spend', min_spend.toString());
    if (max_spend) queryParams.append('max_spend', max_spend.toString());
    if (request_id) queryParams.append('request_id', request_id);
    if (start_date) queryParams.append('start_date', start_date);
    if (end_date) queryParams.append('end_date', end_date);
    if (page) queryParams.append('page', page.toString());
    if (page_size) queryParams.append('page_size', page_size.toString());

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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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

    //message.info("Making spend logs request");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    //message.success("Spend Logs received");
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

    //message.info("Making spend keys request");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    //message.success("Spend Logs received");
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

    //message.info("Making top end users request");

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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    //message.success("Top End users received");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      throw new Error("Network response was not ok");
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

    //message.info("Making top models request");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    //message.success("Top Models received");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    console.log(data);
    //message.success("");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    console.log(data);
    //message.success("");
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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    console.log(data);
    //message.success("Got all users");
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
      const errorData = await response.text();
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
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
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
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
      throw new Error("Network response was not ok");
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
      throw new Error("Network response was not ok");
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
}

export const teamMemberAddCall = async (
  accessToken: string,
  teamId: string,
  formValues: Member // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in teamMemberAddCall:", formValues); // Log the form values before making the API call

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
        member: formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
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

export const teamMemberUpdateCall = async (
  accessToken: string,
  teamId: string,
  formValues: Member // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in teamMemberAddCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/team/member_update`
      : `/team/member_update`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        team_id: teamId,
        role: formValues.role,  
        user_id: formValues.user_id,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
}

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
        ...(formValues.user_email && { user_email: formValues.user_email }),
        ...(formValues.user_id && { user_id: formValues.user_id })
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log("API Response:", data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
}

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
      const errorData = await response.text();
      handleError(errorData);
      console.error("Error response from the server:", errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log("API Response:", data);
    //message.success("User role updated");
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

    //message.info("Predicting spend logs request");

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
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    //message.success("Predicted Logs received");
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
    //message.info("Sending Test Slack alert...");

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
    message.success("Test Slack Alert worked - check your Slack!");
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
    message.success(
      `Test request to ${service} made - check logs/alerts on ${service} to verify`
    );
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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

    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error("Network response was not ok");
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
    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
    message.success("Successfully updated value!");
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
    let url = proxyBaseUrl ? `${proxyBaseUrl}/config/pass_through_endpoint` : `/config/pass_through_endpoint`;

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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
    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
    message.success("Successfully updated value!");
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
    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      const errorData = await response.text();
      handleError(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    message.success("Field reset on proxy");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};

export const deletePassThroughEndpointsCall = async (accessToken: String, endpointId: string) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/config/pass_through_endpoint?endpoint_id=${endpointId}`
      : `/config/pass_through_endpoint${endpointId}`;

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
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

    //message.info("Requesting model data");
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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to call /health:", error);
    throw error;
  }
};

export const getProxyUISettings = async (
  accessToken: String,
) => {
  /**
   * Get all the models user has access to
   */
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/sso/get/ui_settings`
      : `/sso/get/ui_settings`;

    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        [globalLitellmHeaderName]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    //message.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get callbacks:", error);
    throw error;
  }
};



export const getGuardrailsList = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/guardrails/list` : `/guardrails/list`;

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
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log("Guardrails list response:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch guardrails list:", error);
    throw error;
  }
};
