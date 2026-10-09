"use client";

import { Controller, useFormContext, useWatch } from "react-hook-form";
import { Switch } from "@/components/ui/switch";
import { AnalysisKeyPicker } from "./AnalysisKeyPicker";

import { AnalysisAccessFields } from "./AnalysisAccessFields";
import type { WorkerFormInput } from "./workerSchema";

export function WorkerForm({ editingWorker }: { editingWorker: string | null }) {
  const { control } = useFormContext<WorkerFormInput>();
  const useExisting = useWatch({ control, name: "useExisting" });
  return (
    <div className="min-w-0 space-y-5">
      {useExisting ? <AnalysisKeyPicker /> : <AnalysisAccessFields />}
      <details className="text-sm" open={editingWorker ? true : undefined}>
        <summary className="cursor-pointer font-medium">Advanced options</summary>
        <div className="mt-4 space-y-5">
          <label className="flex items-center justify-between gap-4">
            Use an existing virtual key
            <Controller
              control={control}
              name="useExisting"
              render={({ field }) => <Switch checked={field.value} onCheckedChange={field.onChange} />}
            />
          </label>
        </div>
      </details>
    </div>
  );
}
