import { message } from "antd";
import { provider_map } from "../provider_info_helpers";
import { modelCreateCall, Model } from "../networking";


export const handleAddModelSubmit = async (
    formValues: Record<string, any>,
    accessToken: string,
    form: any
  ) => {
    try {
      console.log("handling submit for formValues:", formValues);
      // If model_name is not provided, use provider.toLowerCase() + "/*"
      if (formValues["model"] && formValues["model"].includes("all-wildcard")) {
        const wildcardModel = formValues["custom_llm_provider"].toLowerCase() + "/*";
        formValues["model_name"] = wildcardModel;
        formValues["model"] = wildcardModel; 
      }
      /**
       * For multiple litellm model names - create a separate deployment for each
       * - get the list
       * - iterate through it
       * - create a new deployment for each
       *
       * For single model name -> make it a 1 item list
       */
  
      // get the list of deployments
      let deployments: Array<string> = Array.isArray(formValues["model"])
        ? formValues["model"]
        : [formValues["model"]];
      console.log(`received deployments: ${deployments}`);
      console.log(`received type of deployments: ${typeof deployments}`);
      deployments.forEach(async (litellm_model) => {
        console.log(`litellm_model: ${litellm_model}`);
        const litellmParamsObj: Record<string, any> = {};
        const modelInfoObj: Record<string, any> = {};
        // Iterate through the key-value pairs in formValues
        litellmParamsObj["model"] = litellm_model;
        let modelName: string = "";
        console.log("formValues add deployment:", formValues);
        for (const [key, value] of Object.entries(formValues)) {
          if (value === "") {
            continue;
          }
          if (key == "model_name") {
            modelName = modelName + value;
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
  
          // Check if key is any of the specified API related keys
          else {
            // Add key-value pair to litellm_params dictionary
            litellmParamsObj[key] = value;
          }
        }
  
        const new_model: Model = {
          model_name: modelName,
          litellm_params: litellmParamsObj,
          model_info: modelInfoObj,
        };
  
        const response: any = await modelCreateCall(accessToken, new_model);
  
        console.log(`response for model create call: ${response["data"]}`);
      });
  
      form.resetFields();
    } catch (error) {
      message.error("Failed to create model: " + error, 10);
    }
  };