import { message } from "antd";
import { testConnectionRequest } from "../networking";
import { prepareModelAddRequest } from "./handle_add_model_submit";

export const testModelConnection = async (
  formValues: Record<string, any>,
  accessToken: string,
  testMode: string,
  setConnectionError?: (error: Error | string | null) => void
) => {
  try {
    // Prepare the model data using the existing function
    const result = await prepareModelAddRequest(formValues, accessToken, null);
    
    if (!result) {
      throw new Error("Failed to prepare model data");
    }
    
    const { litellmParamsObj, modelInfoObj } = result;
    
    // Create the request body for the test connection
    const requestBody = {
      ...litellmParamsObj,  // Unfurl the parameters directly
      mode: testMode
    };
    
    // Call the test connection endpoint
    const response = await testConnectionRequest(accessToken, requestBody);
    
    if (response.status === "success") {
      message.success("Connection test successful!");
      // Clear any previous error when successful
      if (setConnectionError) {
        setConnectionError(null);
      }
    } else {
      // Extract the detailed error message from the response
      let errorMessage = response.message || "Unknown error";
      
      // Check if there's a more detailed error in the result
      if (response.result && response.result.error) {
        errorMessage = response.result.error;
      }
      
      if (setConnectionError) {
        setConnectionError(errorMessage);
      } else {
        message.error("Connection test failed: " + errorMessage);
      }
    }
    
    return response;
  } catch (error) {
    console.error("Test connection error:", error);
    
    // Set the error for ConnectionErrorDisplay
    if (setConnectionError) {
      setConnectionError(error);
    } else {
      message.error("Test connection failed: " + error, 10);
    }
    
    return { status: "error", message: error instanceof Error ? error.message : String(error) };
  }
}; 