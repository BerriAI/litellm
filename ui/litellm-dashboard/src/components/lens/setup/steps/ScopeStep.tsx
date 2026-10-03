"use client";
import { useFormContext, useWatch } from "react-hook-form";
import type { InvestigationInput } from "../investigationSchema";

import { Input } from "@/components/ui/input";

import { MatchingActivity } from "../MatchingActivity";

export function ScopeStep({
  step,
  accessToken,
  setPreviewReady,
}: {
  step: number;
  accessToken: string;
  setPreviewReady: (value: boolean) => void;
}) {
  const { control, register } = useFormContext<InvestigationInput>();
  const name = useWatch({ control, name: "name" });
  const manualSelection = useWatch({ control, name: "manualSelection" });
  return (
    <MatchingActivity
      key={step}
      accessToken={accessToken}
      nameField={
        step === 0 ? (
          <label className="grid gap-2 text-sm font-medium">
            Investigation name
            <Input {...register("name")} placeholder="e.g. Support quality" />
          </label>
        ) : undefined
      }
      mode={step === 0 ? "scope" : "activity"}
      onPreviewReady={setPreviewReady}
      manualSelection={manualSelection}
    />
  );
}
