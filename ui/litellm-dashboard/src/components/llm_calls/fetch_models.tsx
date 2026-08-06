import { apiClient } from "@/components/networking";

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

/**
 * Loads models available to the given key.
 *
 * Prefers OpenAI-compatible `/v1/models` (scoped to the key's access) and enriches
 * entries with `mode` from `/model_group/info` when that endpoint is available.
 * Falls back to `/model_group/info` alone if `/v1/models` is empty or fails.
 */
export const fetchAvailableModels = async (accessToken: string): Promise<ModelGroup[]> => {
  const modeByName = new Map<string, string | undefined>();

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

  return [];
};
