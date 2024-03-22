/**
 * Helper file for calls being made to proxy
 */
import { message } from "antd";

const isLocal = process.env.NODE_ENV === "development";
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;

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
        message.error("Failed to parse metadata: " + error);
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    console.log("Form Values after check:", formValues);
    const url = proxyBaseUrl ? `${proxyBaseUrl}/key/generate` : `/key/generate`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userID,
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed to create key: " + errorData);
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

export const userCreateCall = async (
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
        message.error("Failed to parse metadata: " + error);
        throw new Error("Failed to parse metadata: " + error);
      }
    }

    console.log("Form Values after check:", formValues);
    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/new` : `/user/new`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userID,
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed to create key: " + errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        keys: [user_key],
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed to delete key: " + errorData);
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

export const userInfoCall = async (
  accessToken: String,
  userID: String | null,
  userRole: String,
  viewAll: Boolean = false
) => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/user/info` : `/user/info`;
    if (userRole == "App Owner" && userID) {
      url = `${url}?user_id=${userID}`;
    }
    console.log("in userInfoCall viewAll=", viewAll);
    if (viewAll) {
      url = `${url}?view_all=true`;
    }
    //message.info("Requesting user data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log("API Response:", data);
    //message.info("Received user data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const getTotalSpendCall = async (
  accessToken: String,
) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/global/spend` : `/global/spend`;

    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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


export const modelMetricsCall = async (
  accessToken: String,
  userID: String,
  userRole: String
) => {
  /**
   * Get all models on proxy
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/model/metrics` : `/model/metrics`;
    // message.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
  userRole: String
) => {
  /**
   * Get all the models user has access to
   */
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/models` : `/models`;

    //message.info("Requesting model data");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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

export const adminSpendLogsCall = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/logs`
      : `/global/spend/logs`;

    //message.info("Making spend logs request");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
  keyToken: String | null
) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/end_users`
      : `/global/spend/end_users`;

    let body = "";
    if (keyToken) {
      body = JSON.stringify({ api_key: keyToken });
    }
    //message.info("Making top end users request");

    // Define requestOptions with body as an optional property
    const requestOptions: {
      method: string;
      headers: {
        Authorization: string;
        "Content-Type": string;
      };
      body?: string; // The body is optional and might not be present
    } = {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    };

    if (keyToken) {
      requestOptions.body = JSON.stringify({ api_key: keyToken });
    }

    const response = await fetch(url, requestOptions);
    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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

export const adminTopModelsCall = async (accessToken: String) => {
  try {
    let url = proxyBaseUrl
      ? `${proxyBaseUrl}/global/spend/models?limit=5`
      : `/global/spend/models?limit=5`;

    //message.info("Making top models request");
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        keys: keys,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
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
      message.error("Failed to delete key: " + errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed to delete key: " + errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed to delete key: " + errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed to create key: " + errorData);
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

export interface Member {
  role: string;
  user_id: string | null;
  user_email: string | null;
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        team_id: teamId,
        member: formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed to create key: " + errorData);
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

export const userUpdateUserCall = async (
  accessToken: string,
  formValues: any // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in userUpdateUserCall:", formValues); // Log the form values before making the API call

    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/update` : `/user/update`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_role: "proxy_admin_viewer",
        ...formValues, // Include formValues in the request body
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed to create key: " + errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data: requestData,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
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
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.text();
      message.error("Failed Slack Alert test: " + errorData);
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

