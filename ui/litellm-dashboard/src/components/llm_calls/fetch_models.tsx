// fetch_models.ts

import { modelHubCall } from "@/components/networking";

export interface ModelGroup {
  model_group: string;
  mode?: string;
}

interface AvailableModel {
  model_group?: string | null;
  model_name?: string | null;
  id?: string | null;
  mode?: string | null;
}

/**
 * Fetches available models using modelHubCall and formats them for the selection dropdown.
 */
export const fetchAvailableModels = async (accessToken: string): Promise<ModelGroup[]> => {
  try {
    const fetchedModels = await modelHubCall(accessToken);

    if (fetchedModels?.data.length > 0) {
      const models: ModelGroup[] = fetchedModels.data
        .map((item: AvailableModel) => ({
          model_group: item.model_group || item.id || item.model_name || "",
          mode: item.mode || undefined,
        }))
        .filter((model: ModelGroup) => model.model_group !== "");

      models.sort((a, b) => a.model_group.localeCompare(b.model_group));
      return Array.from(new Map(models.map((model) => [model.model_group, model])).values());
    }
    return [];
  } catch (error) {
    console.error("Error fetching model info:", error);
    throw error;
  }
};
