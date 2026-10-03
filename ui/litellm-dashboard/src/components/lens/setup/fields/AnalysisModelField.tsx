"use client";

import { Controller, useFormContext, useWatch } from "react-hook-form";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { analysisModelOptions, type AnalysisModelInfo } from "./analysisModels";
import type { InvestigationInput } from "../investigationSchema";

export function AnalysisModelField({
  models,
  modelDetails,
  modelsLoading,
  modelsError,
  unavailable,
  unsupported,
}: {
  models: string[];
  modelDetails: AnalysisModelInfo[];
  modelsLoading: boolean;
  modelsError?: string;
  unavailable: boolean;
  unsupported: boolean;
}) {
  const { control } = useFormContext<InvestigationInput>();
  const model = useWatch({ control, name: "selectedModel" });
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium">Analysis model</p>
      <Controller
        control={control}
        name="selectedModel"
        render={({ field }) => (
          <SearchSelect
            aria-label="Analysis model"
            options={analysisModelOptions(models, modelDetails)}
            value={field.value ?? ""}
            onValueChange={(value) => field.onChange(value ?? "")}
            placeholder={modelsLoading ? "Loading models…" : "Choose a model"}
            disabled={modelsLoading}
            emptyText="No matching models configured on this gateway"
          />
        )}
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
