// fetch_models.ts

import { excludeProxyWideSentinel } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { modelAvailableCall, modelHubCall } from "@/components/networking";

export interface ModelGroup {
  model_group: string;
  mode?: string;
}

export const fetchAvailableModelsForTeam = async (accessToken: string, teamId: string): Promise<ModelGroup[]> => {
  const response = await modelAvailableCall(accessToken, "", "", false, teamId);
  const modelNames: string[] = (response?.data ?? []).map((model: { id: string }) => model.id);

  return excludeProxyWideSentinel(Array.from(new Set(modelNames)))
    .sort((a, b) => a.localeCompare(b))
    .map((model) => ({ model_group: model }));
};

/**
 * Fetches available models using modelHubCall and formats them for the selection dropdown.
 */
export const fetchAvailableModels = async (accessToken: string): Promise<ModelGroup[]> => {
  try {
    const fetchedModels = await modelHubCall(accessToken);

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
