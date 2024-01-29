/**
 * Helper file for calls being made to proxy
 */

export const keyCreateCall = async (
  proxyBaseUrl: String,
  accessToken: String,
  userID: String
) => {
  try {
    const response = await fetch(`${proxyBaseUrl}/key/generate`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userID,
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
