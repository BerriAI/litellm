// fetch_models.ts

import { modelHubCall } from "../../networking";

export interface ModelGroup {
  model_group: string;
  mode?: string;
}

/**
 * Fetches available models using modelHubCall and formats them for the selection dropdown.
 */
export const fetchAvailableModels = async (accessToken: string): Promise<ModelGroup[]> => {
  try {
    const fetchedModels = await modelHubCall(accessToken);
    console.log("model_info:", fetchedModels);

    if (fetchedModels?.data.length > 0) {
      const models: ModelGroup[] = fetchedModels.data.map((item: any) => ({
        model_group: item.model_group, // Display the model_group to the user
        mode: item?.mode, // Save the mode for auto-selection of endpoint type
      }));

      // Sort models alphabetically by label
      models.sort((a, b) => a.model_group.localeCompare(b.model_group));
      return models;
    }
    return [];
  } catch (error) {
    console.error("Error fetching model info:", error);
    throw error;
  }
};
