import { parseAsArrayOf, parseAsString, useQueryState } from "nuqs";
import { useMemo } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

import { EndpointId, type EndpointIdType } from "./endpoint_config";

export const MAX_COMPARISONS = 3;

const DEFAULT_PANEL_MODELS: string[] = ["", ""];

const ENDPOINT_IDS: readonly EndpointIdType[] = Object.values(EndpointId);

const panelModelsParser = parseAsArrayOf(parseAsString).withDefault(DEFAULT_PANEL_MODELS);

export function resolvePanelModels(urlModels: readonly string[], modelOptions: readonly string[]): string[] {
  const slots = urlModels.length === 0 ? [""] : urlModels.slice(0, MAX_COMPARISONS);
  if (modelOptions.length === 0) return slots;
  return slots.map((model, index) =>
    modelOptions.includes(model) ? model : modelOptions[index % modelOptions.length],
  );
}

export function useCompareUrlState(modelOptions: readonly string[]) {
  const [urlModels, setUrlModels] = useQueryState("cmp_models", panelModelsParser);
  const [endpoint, setEndpoint] = useUrlTab(ENDPOINT_IDS, EndpointId.CHAT_COMPLETIONS, "cmp_endpoint");
  const panelModels = useMemo(() => resolvePanelModels(urlModels, modelOptions), [urlModels, modelOptions]);

  const setPanelModel = (index: number, model: string) =>
    void setUrlModels(panelModels.map((current, i) => (i === index ? model : current)));

  const addPanel = (model: string) => void setUrlModels([...panelModels, model]);

  const removePanel = (index: number) => void setUrlModels(panelModels.filter((_, i) => i !== index));

  return { endpoint, panelModels, setEndpoint, setPanelModel, addPanel, removePanel };
}
