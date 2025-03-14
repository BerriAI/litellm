import { message } from "antd";
import { testConnectionRequest } from "../networking";
import { prepareModelAddRequest } from "./handle_add_model_submit";

export const testModelConnection = async (
  formValues: Record<string, any>,
  accessToken: string,
  testMode: string,
  setConnectionError?: (error: Error | string | null, rawRequest?: any, rawResponse?: any) => void
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
      ...litellmParamsObj,
      mode: testMode
    };
    
    console.log("Request Body:", requestBody); // Debugging log

    // Call the test connection endpoint
    const response = await testConnectionRequest(accessToken, requestBody);
    
    console.log("Response:", response); // Debugging log

    if (response.status === "success") {
      message.success("Connection test successful!");
      if (setConnectionError) {
        setConnectionError(null);
      }
    } else {
      let errorMessage = response.message || "Unknown error";
      if (response.result && response.result.error) {
        errorMessage = response.result.error;
      }
      
      if (setConnectionError) {
        setConnectionError(errorMessage, requestBody, response.result.raw_request_typed_dict);
      } else {
        message.error("Connection test failed: " + errorMessage);
      }
    }
    
    return response;
  } catch (error) {
    console.error("Test connection error:", error);
    
    if (setConnectionError) {
      setConnectionError(error, requestBody, null);
    } else {
      message.error("Test connection failed: " + error, 10);
    }
    
    return { status: "error", message: error instanceof Error ? error.message : String(error) };
  }
}; 