import { parseAsArrayOf, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useMemo } from "react";

import { EndpointId, type EndpointIdType } from "./endpoint_config";

export const MAX_COMPARISONS = 3;

const DEFAULT_PANEL_MODELS: string[] = ["", ""];

const compareUrlParsers = {
  cmp_models: parseAsArrayOf(parseAsString).withDefault(DEFAULT_PANEL_MODELS),
  cmp_endpoint: parseAsStringLiteral(Object.values(EndpointId)).withDefault(EndpointId.CHAT_COMPLETIONS),
};

export function resolvePanelModels(urlModels: readonly string[], modelOptions: readonly string[]): string[] {
  const slots = urlModels.length === 0 ? [""] : urlModels.slice(0, MAX_COMPARISONS);
  if (modelOptions.length === 0) return slots;
  return slots.map((model, index) =>
    modelOptions.includes(model) ? model : modelOptions[index % modelOptions.length],
  );
}

export function useCompareUrlState(modelOptions: readonly string[]) {
  const [{ cmp_models: urlModels, cmp_endpoint: endpoint }, setUrl] = useQueryStates(compareUrlParsers);
  const panelModels = useMemo(() => resolvePanelModels(urlModels, modelOptions), [urlModels, modelOptions]);

  const setEndpoint = (next: EndpointIdType) => void setUrl({ cmp_endpoint: next });

  const setPanelModel = (index: number, model: string) =>
    void setUrl({ cmp_models: panelModels.map((current, i) => (i === index ? model : current)) });

  const addPanel = (model: string) => void setUrl({ cmp_models: [...panelModels, model] });

  const removePanel = (index: number) => void setUrl({ cmp_models: panelModels.filter((_, i) => i !== index) });

  return { endpoint, panelModels, setEndpoint, setPanelModel, addPanel, removePanel };
}
