import type { AnalysisModelInfo } from "../../model/types";
import type { AnalysisModels } from "./useAnalysisModels";

export interface ModelGate {
  readonly modelValid: boolean;
  readonly unavailable: boolean;
  readonly unsupported: boolean;
}

/** An edit may keep its saved model while the model list is loading or failing; a new run needs a verified one. */
export function modelGate(models: AnalysisModels, model: string, preservingSavedModel: boolean): ModelGate {
  const unsupported = models.modelDetails.some((m) => m.model_group === model && m.mode && m.mode !== "chat");
  const modelsReady = !models.modelsLoading && !models.modelsError;
  const unavailable = !!model && modelsReady && !models.models.includes(model);
  const supported = !unsupported && !unavailable;
  const verified = modelsReady || preservingSavedModel;
  return { modelValid: !!model && supported && verified, unavailable, unsupported };
}

export function analysisModelOptions(models: string[], details: AnalysisModelInfo[]) {
  return [...new Set(models)].sort().map((name) => {
    const info = details.find((item) => item.model_group === name);
    const capability = () => {
      if (info?.mode && info.mode !== "chat") return `${info.mode}: not suitable for Lens`;
      if (info?.supported_openai_params?.includes("response_format")) return "JSON output supported";
      return "JSON output support unverified";
    };
    return {
      value: name,
      label: name,
      sublabel: [info?.providers.join(", "), capability()].filter(Boolean).join(" · "),
    };
  });
}
