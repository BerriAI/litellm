/**
 * Helper file for calls being made to proxy
 */
import { message } from 'antd';

// const proxyBaseUrl = null;
const proxyBaseUrl = "http://localhost:4000" // http://localhost:4000

export const keyCreateCall = async (
  accessToken: string,
  userID: string,
  formValues: Record<string, any>, // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in keyCreateCall:", formValues); // Log the form values before making the API call    
    
    // if formValues.metadata is not undefined, make it a valid dict 
    if (formValues.metadata) {
      // if there's an exception JSON.parse, show it in the message
      try {
        formValues.metadata = JSON.parse(formValues.metadata);
      } catch (error) {
        message.error("Failed to parse metadata: " + error);
      }
    }
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


export const keyDeleteCall = async (
  accessToken: String,
  user_key: String
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/key/delete` : `/key/delete`;
    console.log("in keyDeleteCall:", user_key)
    
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
  userID: String
) => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/info` : `/user/info`;
    console.log("in userInfoCall:", url)
    const response = await fetch(
      `${url}/?user_id=${userID}`,
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      const errorData = await response.text();
      message.error(errorData);
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log(data);
    return data;
    // Handle success - you might want to update some state or UI based on the created key
  } catch (error) {
    console.error("Failed to create key:", error);
    throw error;
  }
};
