"use client";

import { SearchSelect } from "@/components/shared/SearchSelect";
import { analysisModelOptions, type AnalysisModelInfo } from "./analysisModels";

export function AnalysisModelField({
  models,
  modelDetails,
  model,
  setModel,
  modelsLoading,
  modelsError,
  unavailable,
  unsupported,
}: {
  models: string[];
  modelDetails: AnalysisModelInfo[];
  model: string;
  setModel: (model: string) => void;
  modelsLoading: boolean;
  modelsError?: string;
  unavailable: boolean;
  unsupported: boolean;
}) {
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium">Analysis model</p>
      <SearchSelect
        aria-label="Analysis model"
        options={analysisModelOptions(models, modelDetails)}
        value={model}
        onValueChange={(value) => setModel(value ?? "")}
        placeholder={modelsLoading ? "Loading models…" : "Choose a model"}
        disabled={modelsLoading}
        emptyText="No matching models configured on this gateway"
      />
      {modelsError && (
        <p role="alert" className="text-sm text-destructive">
          Could not load models: {modelsError}
        </p>
      )}
      {unavailable && (
        <p role="alert" className="text-sm text-destructive">
          {model} is no longer available. Choose another analysis model.
        </p>
      )}
      {unsupported && (
        <p role="alert" className="text-sm text-destructive">
          Choose a chat model that supports JSON output.
        </p>
      )}
    </div>
  );
}
