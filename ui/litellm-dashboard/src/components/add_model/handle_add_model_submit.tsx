import { message } from "antd";
import { provider_map, Providers } from "../provider_info_helpers";
import { modelCreateCall, Model, testConnectionRequest } from "../networking";
import React, { useState } from 'react';
import ConnectionErrorDisplay from './ConnectionErrorDisplay';

export const prepareModelAddRequest = async (
    formValues: Record<string, any>,
    accessToken: string,
    form: any,
  ) => {
    try {
      console.log("handling submit for formValues:", formValues);


      // Get model mappings and safely remove from formValues
      const modelMappings = formValues["model_mappings"] || [];
      if ("model_mappings" in formValues) {
        delete formValues["model_mappings"];
      }
      
      
      // Handle wildcard case
      if (formValues["model"] && formValues["model"].includes("all-wildcard")) {
        const customProvider: Providers = formValues["custom_llm_provider"];
        const litellm_custom_provider = provider_map[customProvider as keyof typeof Providers];
        const wildcardModel = litellm_custom_provider + "/*";
        formValues["model_name"] = wildcardModel;
        modelMappings.push({
          public_name: wildcardModel,
          litellm_model: wildcardModel,
        });
        formValues["model"] = wildcardModel; 
      }

      // Create a deployment for each mapping
      for (const mapping of modelMappings) {
        const litellmParamsObj: Record<string, any> = {};
        const modelInfoObj: Record<string, any> = {};
        
        // Set the model name and litellm model from the mapping
        const modelName = mapping.public_name;
        litellmParamsObj["model"] = mapping.litellm_model;

        // Handle pricing conversion before processing other fields
        if (formValues.input_cost_per_token) {
          formValues.input_cost_per_token = Number(formValues.input_cost_per_token) / 1000000;
        }
        if (formValues.output_cost_per_token) {
          formValues.output_cost_per_token = Number(formValues.output_cost_per_token) / 1000000;
        }
        // Keep input_cost_per_second as is, no conversion needed
        
        // Iterate through the key-value pairs in formValues
        litellmParamsObj["model"] = mapping.litellm_model;
        console.log("formValues add deployment:", formValues);
        for (const [key, value] of Object.entries(formValues)) {
          if (value === "") {
            continue;
          }
          // Skip the custom_pricing and pricing_model fields as they're only used for UI control
          if (key === 'custom_pricing' || key === 'pricing_model') {
            continue;
          }
          if (key == "model_name") {
            litellmParamsObj["model"] = value;
          } else if (key == "custom_llm_provider") {
            console.log("custom_llm_provider:", value);
            const mappingResult = provider_map[value]; // Get the corresponding value from the mapping
            litellmParamsObj["custom_llm_provider"] = mappingResult;
            console.log("custom_llm_provider mappingResult:", mappingResult);
          } else if (key == "model") {
            continue;
          }
  
          // Check if key is "base_model"
          else if (key === "base_model") {
            // Add key-value pair to model_info dictionary
            modelInfoObj[key] = value;
          }
          else if (key === "team_id") {
            modelInfoObj["team_id"] = value;
          }
          else if (key === "custom_model_name") {
            litellmParamsObj["model"] = value;
          } else if (key == "litellm_extra_params") {
            console.log("litellm_extra_params:", value);
            let litellmExtraParams = {};
            if (value && value != undefined) {
              try {
                litellmExtraParams = JSON.parse(value);
              } catch (error) {
                message.error(
                  "Failed to parse LiteLLM Extra Params: " + error,
                  10
                );
                throw new Error("Failed to parse litellm_extra_params: " + error);
              }
              for (const [key, value] of Object.entries(litellmExtraParams)) {
                litellmParamsObj[key] = value;
              }
            }
          } else if (key == "model_info_params") {
            console.log("model_info_params:", value);
            let modelInfoParams = {};
            if (value && value != undefined) {
              try {
                modelInfoParams = JSON.parse(value);
              } catch (error) {
                message.error(
                  "Failed to parse LiteLLM Extra Params: " + error,
                  10
                );
                throw new Error("Failed to parse litellm_extra_params: " + error);
              }
              for (const [key, value] of Object.entries(modelInfoParams)) {
                modelInfoObj[key] = value;
              }
            }
          }
  
          // Handle the pricing fields
          else if (key === "input_cost_per_token" || 
                  key === "output_cost_per_token" || 
                  key === "input_cost_per_second") {
            if (value) {
              litellmParamsObj[key] = Number(value);
            }
            continue;
          }
  
          // Check if key is any of the specified API related keys
          else {
            // Add key-value pair to litellm_params dictionary
            litellmParamsObj[key] = value;
          }
        }

        return { litellmParamsObj, modelInfoObj, modelName };
      }
    } catch (error) {
      message.error("Failed to create model: " + error, 10);
    }
  };

export const handleAddModelSubmit = async (
    accessToken: string,
    form: any,
    callback?: () => void,
  ) => {
    try {
      const formValues = form.getFieldsValue();
      const result = await prepareModelAddRequest(formValues, accessToken, form);
      
      if (!result) {
        return; // Exit if preparation failed
      }
      
      const { litellmParamsObj, modelInfoObj, modelName } = result;
      
      const new_model: Model = {
        model_name: modelName,
        litellm_params: litellmParamsObj,
        model_info: modelInfoObj,
      };
      
      const response: any = await modelCreateCall(accessToken, new_model);
      console.log(`response for model create call: ${response["data"]}`);
      
      callback && callback();
      form.resetFields();
      
      message.success("Model added successfully");
    } catch (error) {
      message.error("Failed to add model: " + error, 10);
    }
  };

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
      // Set the error for ConnectionErrorDisplay instead of showing a message
      const errorMessage = response.message || "Unknown error";
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
    
    throw error;
  }
};

     
