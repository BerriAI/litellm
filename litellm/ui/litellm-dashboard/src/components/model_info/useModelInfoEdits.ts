import { useEffect, useRef, useState } from "react";

import { editedModelInfoPricing, parseModelExtraParams } from "../../utils/modelEditParams";

type SavedModelInfo = { fetchedAt: number; value: Record<string, unknown> };

/**
 * Tracks what the Model Info JSON box showed when editing started, so only rates the operator changed
 * become overrides, and keeps showing a just-saved Model Info until the model is fetched again.
 */
export const useModelInfoEdits = <T extends { model_info: unknown }>(
  fetchedModel: T | null | undefined,
  fetchedAt: number,
  setIsEditing: (isEditing: boolean) => void,
) => {
  const [saved, setSaved] = useState<SavedModelInfo | null>(null);
  const shownAtEditStart = useRef<Record<string, unknown>>({});
  const latestFetchedAt = useRef(fetchedAt);
  useEffect(() => {
    latestFetchedAt.current = fetchedAt;
  }, [fetchedAt]);
  const modelData =
    fetchedModel && saved !== null && saved.fetchedAt === fetchedAt
      ? { ...fetchedModel, model_info: saved.value as T["model_info"] }
      : fetchedModel;

  return {
    modelData,
    startEditing: () => {
      shownAtEditStart.current = { ...(modelData?.model_info as Record<string, unknown> | undefined) };
      setIsEditing(true);
    },
    pricing: (
      modelInfoJson: string | undefined,
      editedParams: Record<string, unknown>,
      loadedModel: { litellm_params?: Record<string, unknown> } | null,
    ) =>
      editedModelInfoPricing(
        parseModelExtraParams(modelInfoJson) ?? {},
        shownAtEditStart.current,
        editedParams,
        loadedModel?.litellm_params ?? {},
      ),
    rememberSaved: (value: Record<string, unknown>) => setSaved({ fetchedAt: latestFetchedAt.current, value }),
  };
};
