"use client";

import { Controller, useFormContext, useWatch } from "react-hook-form";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { analysisModelOptions, type ModelGate } from "./analysisModels";
import type { AnalysisModels } from "./useAnalysisModels";
import type { InvestigationInput } from "../investigationSchema";

export interface AnalysisModelFieldProps {
  readonly models: AnalysisModels;
  readonly gate: ModelGate;
}

export function AnalysisModelField({ models, gate }: AnalysisModelFieldProps) {
  const { control } = useFormContext<InvestigationInput>();
  const model = useWatch({ control, name: "selectedModel" });
  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">Analysis model</span>
      <Controller
        control={control}
        name="selectedModel"
        render={({ field }) => (
          <SearchSelect
            aria-label="Analysis model"
            options={analysisModelOptions(models.models, models.modelDetails)}
            value={field.value ?? ""}
            onValueChange={(value) => field.onChange(value ?? "")}
            placeholder={models.modelsLoading ? "Loading models…" : "Choose a model"}
            disabled={models.modelsLoading}
            emptyText="No matching models configured on this gateway"
          />
        )}
      />
      {models.modelsError && (
        <p role="alert" className="text-sm text-destructive">
          Could not load models: {models.modelsError}
        </p>
      )}
      {gate.unavailable && (
        <p role="alert" className="text-sm text-destructive">
          {model} is no longer available. Choose another analysis model.
        </p>
      )}
      {gate.unsupported && (
        <p role="alert" className="text-sm text-destructive">
          Choose a chat model that supports JSON output.
        </p>
      )}
    </div>
  );
}
