import { message } from "antd";
import { modelCreateCall, Model } from "../networking";

export const handleAddAutoRouterSubmit = async (
  values: any,
  accessToken: string,
  form: any,
  callback?: () => void,
) => {
  try {
    console.log("handling auto router submit for formValues:", values);

    // Create auto router configuration
    const autoRouterConfig = {
      model_name: values.auto_router_name,
      litellm_params: {
        model: `auto_router/${values.auto_router_name}`,
        auto_router_config: values.auto_router_config, // Use built JSON config instead of file path
        auto_router_default_model: values.auto_router_default_model,
      },
      model_info: {},
    };

    // Add optional embedding model if provided
    if (values.auto_router_embedding_model) {
      autoRouterConfig.litellm_params.auto_router_embedding_model = values.auto_router_embedding_model;
    }

    // Add team information if provided
    if (values.team_id) {
      autoRouterConfig.model_info.team_id = values.team_id;
    }

    // Add model access groups if provided
    if (values.model_access_group && values.model_access_group.length > 0) {
      autoRouterConfig.model_info.access_groups = values.model_access_group;
    }

    console.log("Auto router configuration to be created:", autoRouterConfig);

    // Create the auto router using the same model creation endpoint
    const response: any = await modelCreateCall(accessToken, autoRouterConfig as Model);
    console.log(`response for auto router create call: ${response["data"]}`);

    message.success("Auto router added successfully!");
    
    // Call the callback function if provided (usually to refresh the model list)
    callback && callback();
    
    // Reset the form
    form.resetFields();
    
  } catch (error) {
    console.error("Failed to add auto router:", error);
    message.error("Failed to add auto router: " + error, 10);
  }
}; 