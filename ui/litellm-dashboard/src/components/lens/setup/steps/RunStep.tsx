"use client";
import { Controller, useFormContext } from "react-hook-form";
import type { InvestigationInput } from "../investigationSchema";

import { Input } from "@/components/ui/input";

import { DurationInput } from "@/components/shared/DurationInput";
import { type AnalysisModelInfo } from "../fields/analysisModels";

import { AnalysisModelField } from "../fields/AnalysisModelField";
export function RunStep({
  modelValid,
  models,
  modelDetails,
  model,
  setModel,
  modelsLoading,
  modelsError,
  unavailable,
  unsupported,
}: {
  modelValid: boolean;
  models: string[];
  modelDetails: AnalysisModelInfo[];
  model: string;
  setModel: (model: string) => void;
  modelsLoading: boolean;
  modelsError?: string;
  unavailable: boolean;
  unsupported: boolean;
}) {
  const { watch, setValue, control } = useFormContext<InvestigationInput>();
  const { selection, manualSelection, budget, repeat } = watch();
  return (
    <>
      <details open={!modelValid || undefined}>
        <summary className="cursor-pointer text-sm font-medium">Advanced options</summary>
        <div className="mt-4 space-y-5">
          <AnalysisModelField
            models={models}
            modelDetails={modelDetails}
            model={model}
            setModel={setModel}
            modelsLoading={modelsLoading}
            modelsError={modelsError}
            unavailable={unavailable}
            unsupported={unsupported}
          />
          <div className="grid gap-5 sm:grid-cols-2">
            <label className="grid content-start gap-2 text-sm font-medium">
              Maximum runs (optional)
              <Input
                type="number"
                min="1"
                placeholder="No limit"
                value={selection.sample_size ?? ""}
                onChange={(event) =>
                  setValue("selection.sample_size", event.target.value ? Number(event.target.value) : null)
                }
              />
            </label>
            <label className="flex items-center gap-2 text-sm font-medium">
              <input
                type="checkbox"
                checked={manualSelection}
                onChange={(event) => {
                  setValue("manualSelection", event.target.checked);
                  setValue("selection.execution_ids", []);
                }}
              />
              Choose individual runs
            </label>
          </div>
          <div className="grid gap-5 sm:grid-cols-2">
            <label className="grid content-start gap-2 text-sm font-medium">
              Monthly limit (USD)
              <Input
                type="number"
                min="0.01"
                max="100000"
                step="1"
                value={budget}
                onChange={(e) => setValue("budget", Number(e.target.value))}
              />
            </label>
            <div className="space-y-3">
              <label className="flex items-center gap-2 text-sm font-medium">
                <input
                  type="checkbox"
                  checked={repeat}
                  onChange={(e) => setValue("repeat", e.target.checked)}
                  className="size-4 rounded border-input accent-foreground"
                />
                Repeat this investigation
              </label>
              {repeat && (
                <Controller
                  control={control}
                  name="interval"
                  render={({ field }) => (
                    <DurationInput
                      label="Repeat every"
                      value={field.value}
                      onChange={field.onChange}
                      base="minutes"
                      max={10080}
                    />
                  )}
                />
              )}
            </div>
          </div>
        </div>
      </details>
    </>
  );
}
