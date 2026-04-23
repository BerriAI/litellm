/**
 * Utility function to transform raw model data into the format expected by UI components
 * This creates a new transformed data object without mutating the original
 */
export const transformModelData = (rawModelData: any, getProviderFromModel: (model: string) => string) => {
  if (!rawModelData?.data) return { data: [] };

  // Deep copy the data to avoid mutating the original
  const transformedData = JSON.parse(JSON.stringify(rawModelData.data));

  for (let i = 0; i < transformedData.length; i++) {
    let curr_model = transformedData[i];
    let litellm_model_name = curr_model?.litellm_params?.model;
    let custom_llm_provider = curr_model?.litellm_params?.custom_llm_provider;
    let model_info = curr_model?.model_info;

    let provider = "";
    let input_cost: any = null;
    let output_cost: any = null;
    let max_tokens = "Undefined";
    let max_input_tokens = "Undefined";
    let cleanedLitellmParams = {};

    // Check if litellm_model_name is null or undefined
    if (litellm_model_name) {
      // Split litellm_model_name based on "/"
      let splitModel = litellm_model_name.split("/");

      // Get the first element in the split
      let firstElement = splitModel[0];

      // If there is only one element, default provider to openai
      provider = custom_llm_provider;
      if (!provider) {
        provider = splitModel.length === 1 ? getProviderFromModel(litellm_model_name) : firstElement;
      }
    } else {
      // litellm_model_name is null or undefined, default provider to openai
      provider = "-";
    }

    if (model_info) {
      input_cost = model_info?.input_cost_per_token;
      output_cost = model_info?.output_cost_per_token;
      max_tokens = model_info?.max_tokens;
      max_input_tokens = model_info?.max_input_tokens;
    }

    if (curr_model?.litellm_params) {
      cleanedLitellmParams = Object.fromEntries(
        Object.entries(curr_model?.litellm_params).filter(([key]) => key !== "model" && key !== "api_base"),
      );
    }

    transformedData[i].provider = provider;
    transformedData[i].input_cost = input_cost;
    transformedData[i].output_cost = output_cost;
    transformedData[i].litellm_model_name = litellm_model_name;

    // Convert Cost in terms of Cost per 1M tokens
    if (transformedData[i].input_cost != null) {
      transformedData[i].input_cost = (Number(transformedData[i].input_cost) * 1000000).toFixed(2);
    }

    if (transformedData[i].output_cost != null) {
      transformedData[i].output_cost = (Number(transformedData[i].output_cost) * 1000000).toFixed(2);
    }

    transformedData[i].max_tokens = max_tokens;
    transformedData[i].max_input_tokens = max_input_tokens;
    transformedData[i].api_base = curr_model?.litellm_params?.api_base;
    transformedData[i].cleanedLitellmParams = cleanedLitellmParams;
  }

  return { data: transformedData };
};
