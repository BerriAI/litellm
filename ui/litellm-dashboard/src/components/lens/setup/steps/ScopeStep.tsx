"use client";
import { useFormContext } from "react-hook-form";
import type { InvestigationInput } from "../investigationSchema";

import { Input } from "@/components/ui/input";

import { MatchingActivity } from "../MatchingActivity";
import type { ActivitySelection } from "../../model/types";

export function ScopeStep({
  step,
  changeSelection,
  accessToken,
  setPreviewReady,
}: {
  step: number;
  changeSelection: (value: ActivitySelection) => void;
  accessToken: string;
  setPreviewReady: (value: boolean) => void;
}) {
  const { watch, setValue } = useFormContext<InvestigationInput>();
  const { name, selection, manualSelection } = watch();
  const setName = (name: string) => setValue("name", name);
  return (
    <MatchingActivity
      key={step}
      value={selection}
      onChange={changeSelection}
      accessToken={accessToken}
      nameField={
        step === 0 ? (
          <label className="grid gap-2 text-sm font-medium">
            Investigation name
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Support quality"
              maxLength={100}
            />
          </label>
        ) : undefined
      }
      mode={step === 0 ? "scope" : "activity"}
      onPreviewReady={setPreviewReady}
      manualSelection={manualSelection}
    />
  );
}
