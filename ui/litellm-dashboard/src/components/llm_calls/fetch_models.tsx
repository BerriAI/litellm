// fetch_models.ts

import { excludeProxyWideSentinel } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { modelAvailableCall, modelHubCall } from "@/components/networking";

export interface ModelGroup {
  model_group: string;
  mode?: string;
  supports_reasoning?: boolean;
  supported_reasoning_efforts?: string[];
}

interface AvailableModel {
  model_group?: string | null;
  model_name?: string | null;
  id?: string | null;
  mode?: string | null;
  supports_reasoning?: boolean | null;
  supported_reasoning_efforts?: string[] | null;
}

const toModelGroup = (item: AvailableModel): ModelGroup => {
  const groupName = (item.model_group || item.id || item.model_name) ?? "";
  return {
    model_group: groupName,
    ...(item.mode && { mode: item.mode }),
    ...(item.supports_reasoning === true && { supports_reasoning: true }),
    ...(item.supported_reasoning_efforts && { supported_reasoning_efforts: item.supported_reasoning_efforts }),
  };
};

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
    const fetchedData: unknown = fetchedModels?.data;
    const models: ModelGroup[] = (Array.isArray(fetchedData) ? fetchedData : [])
      .map(toModelGroup)
      .filter((model: ModelGroup) => model.model_group !== "")
      .sort((a: ModelGroup, b: ModelGroup) => a.model_group.localeCompare(b.model_group));
    return Array.from(new Map(models.map((model) => [model.model_group, model])).values());
  } catch (error) {
    console.error("Error fetching model info:", error);
    throw error;
  }
};
