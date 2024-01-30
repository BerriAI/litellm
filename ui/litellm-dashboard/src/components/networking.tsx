/**
 * Helper file for calls being made to proxy
 */

export const keyCreateCall = async (
  proxyBaseUrl: string,
  accessToken: string,
  userID: string,
  formValues: Record<string, any> // Assuming formValues is an object
) => {
  try {
    console.log("Form Values in keyCreateCall:", formValues); // Log the form values before making the API call
    const response = await fetch(`${proxyBaseUrl}/key/generate`, {
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
      const errorData = await response.json();
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
  proxyBaseUrl: String,
  accessToken: String,
  user_key: String
) => {
  try {
    const response = await fetch(`${proxyBaseUrl}/key/delete`, {
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

export const userInfoCall = async (
  proxyBaseUrl: String,
  accessToken: String,
  userID: String
) => {
  try {
    const response = await fetch(
      `${proxyBaseUrl}/user/info?user_id=${userID}`,
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
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
