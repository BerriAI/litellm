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

const toModelGroups = (fetchedData: unknown): ModelGroup[] =>
  (Array.isArray(fetchedData) ? fetchedData : [])
    .map(toModelGroup)
    .filter((model: ModelGroup) => model.model_group !== "")
    .sort((a: ModelGroup, b: ModelGroup) => a.model_group.localeCompare(b.model_group));

const dedupeByGroup = (models: ModelGroup[]): ModelGroup[] =>
  Array.from(new Map(models.map((model) => [model.model_group, model])).values());

export const fetchAvailableModelsForTeam = async (accessToken: string, teamId: string): Promise<ModelGroup[]> => {
  const response = await modelAvailableCall(accessToken, "", "", false, teamId);
  const modelNames: string[] = (response?.data ?? []).map((model: { id: string }) => model.id);

  return excludeProxyWideSentinel(Array.from(new Set(modelNames)))
    .sort((a, b) => a.localeCompare(b))
    .map((model) => ({ model_group: model }));
};

/**
 * Fetches the models the given credentials can call, for the selection dropdown.
 *
 * Primary source is /model_group/info (modelHubCall), which carries capability
 * fields like supported_reasoning_efforts. That route returns an empty list for
 * non-admin UI sessions (#38534), so when it comes back empty or fails we fall
 * back to /models — the canonical list (same route as GET /v1/models) that
 * works for both UI session tokens and virtual keys.
 */
export const fetchAvailableModels = async (accessToken: string): Promise<ModelGroup[]> => {
  let models: ModelGroup[] = [];
  try {
    const fetchedModels = await modelHubCall(accessToken);
    models = toModelGroups(fetchedModels?.data);
  } catch (error) {
    console.error("Error fetching model info:", error);
  }

  if (models.length > 0) {
    return dedupeByGroup(models);
  }

  const response = await modelAvailableCall(accessToken, "", "");
  // Same permission sentinel excludeProxyWideSentinel filters elsewhere: it is
  // a marker, not a selectable model.
  const selectable = new Set(
    excludeProxyWideSentinel(
      (Array.isArray(response?.data) ? response.data : []).map(
        (model: AvailableModel) => (model.model_group || model.id || model.model_name) ?? "",
      ),
    ),
  );
  return dedupeByGroup(toModelGroups(response?.data).filter((model) => selectable.has(model.model_group)));
};
