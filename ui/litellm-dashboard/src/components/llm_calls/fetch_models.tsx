import { excludeProxyWideSentinel } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { apiClient, modelAvailableCall } from "@/components/networking";

export interface ModelGroup {
  model_group: string;
  mode?: string;
}

interface ModelGroupInfoItem {
  model_group?: string | null;
  model_name?: string | null;
  id?: string | null;
  mode?: string | null;
}

interface OpenAIModelItem {
  id?: string | null;
}

const modelNameFromGroupItem = (item: ModelGroupInfoItem): string =>
  (item.model_group || item.id || item.model_name || "").trim();

const dedupeAndSort = (models: ModelGroup[]): ModelGroup[] => {
  const unique = Array.from(new Map(models.map((model) => [model.model_group, model])).values());
  unique.sort((a, b) => a.model_group.localeCompare(b.model_group));
  return unique;
};

export const fetchAvailableModelsForTeam = async (accessToken: string, teamId: string): Promise<ModelGroup[]> => {
  const response = await modelAvailableCall(accessToken, "", "", false, teamId);
  const modelNames: string[] = (response?.data ?? []).map((model: { id: string }) => model.id);

  return excludeProxyWideSentinel(Array.from(new Set(modelNames)))
    .sort((a, b) => a.localeCompare(b))
    .map((model) => ({ model_group: model }));
};

export const fetchAvailableModels = async (accessToken: string): Promise<ModelGroup[]> => {
  const modeByName = new Map<string, string | undefined>();
  let groupInfoError: unknown;
  let listModelsError: unknown;

  try {
    const groupInfo = await apiClient.get<{ data?: ModelGroupInfoItem[] }>("/model_group/info", {
      accessToken,
    });
    for (const item of groupInfo?.data ?? []) {
      const name = modelNameFromGroupItem(item);
      if (name) {
        modeByName.set(name, item.mode || undefined);
      }
    }
  } catch (error) {
    groupInfoError = error;
    console.error("Error fetching model group info:", error);
  }

  try {
    const listed = await apiClient.get<{ data?: OpenAIModelItem[] }>("/v1/models", {
      accessToken,
    });
    const fromList = (listed?.data ?? [])
      .map((item) => {
        const name = (item.id || "").trim();
        if (!name) {
          return null;
        }
        return {
          model_group: name,
          mode: modeByName.get(name),
        } satisfies ModelGroup;
      })
      .filter((model): model is ModelGroup => model != null);

    if (fromList.length > 0) {
      return dedupeAndSort(fromList);
    }
  } catch (error) {
    listModelsError = error;
    console.error("Error fetching /v1/models:", error);
  }

  if (modeByName.size > 0) {
    return dedupeAndSort(
      Array.from(modeByName.entries()).map(([model_group, mode]) => ({
        model_group,
        mode,
      })),
    );
  }

  if (groupInfoError && listModelsError) {
    throw listModelsError instanceof Error
      ? listModelsError
      : new Error("Failed to load models from /v1/models and /model_group/info");
  }

  return [];
};
