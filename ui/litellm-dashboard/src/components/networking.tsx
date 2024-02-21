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
    message.info("Making key delete request");
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
    message.success("API Key Deleted");
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
      url = `${url}/?user_id=${userID}`;
    }
    console.log("in userInfoCall viewAll=", viewAll);
    if (viewAll) {
      url = `${url}/?view_all=true`;
    }
    message.info("Requesting user data");
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
    message.info("Received user data");
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
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/v2/model/info` : `/v2/model/info`;

    message.info("Requesting model data");
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
    message.info("Received model data");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};

export const keySpendLogsCall = async (accessToken: String, token: String) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/spend/logs` : `/spend/logs`;
    console.log("in keySpendLogsCall:", url);
    const response = await fetch(`${url}/?api_key=${token}`, {
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
    let url = proxyBaseUrl ? `${proxyBaseUrl}/spend/logs` : `/spend/logs`;
    if (userRole == "App Owner") {
      url = `${url}/?user_id=${userID}&start_date=${startTime}&end_date=${endTime}`;
    } else {
      url = `${url}/?start_date=${startTime}&end_date=${endTime}`;
    }
    message.info("Making spend logs request");
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
    message.success("Spend Logs received");
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
    const response = await fetch(`${url}/?user_id=${userID}`, {
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




export const userRequestModelCall = async (accessToken: String, model: String, UserID: String, justification: String) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/request_model` : `/user/request_model`;
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
    message.success("");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};


export const userGetRequesedtModelsCall = async (accessToken: String) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/get_requests` : `/user/get_requests`;
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
    message.success("");
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to get requested models:", error);
    throw error;
  }
};