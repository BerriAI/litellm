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

/**
 * /models carries no capability metadata, so the team-allowed names are joined against
 * /model_group/info; a name without a group entry keeps every capability field absent (unknown).
 */
export const fetchAvailableModelsForTeam = async (accessToken: string, teamId: string): Promise<ModelGroup[]> => {
  const [response, groups] = await Promise.all([
    modelAvailableCall(accessToken, "", "", false, teamId),
    fetchAvailableModels(accessToken).catch(() => [] as ModelGroup[]),
  ]);
  const byGroup = new Map(groups.map((group) => [group.model_group, group]));
  const modelNames: string[] = (response?.data ?? []).map((model: { id: string }) => model.id);

  return excludeProxyWideSentinel(Array.from(new Set(modelNames)))
    .sort((a, b) => a.localeCompare(b))
    .map((model) => byGroup.get(model) ?? { model_group: model });
};

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
          supports_reasoning: item.supports_reasoning === true || undefined,
          supported_reasoning_efforts: item.supported_reasoning_efforts ?? undefined,
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
