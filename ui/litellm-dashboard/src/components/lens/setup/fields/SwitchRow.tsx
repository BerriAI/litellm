"use client";

import { useId } from "react";
import { Switch } from "@/components/ui/switch";

export interface SwitchRowProps {
  readonly label: string;
  readonly description: string;
  readonly checked: boolean;
  readonly onCheckedChange: (checked: boolean) => void;
}

export function SwitchRow({ label, description, checked, onCheckedChange }: SwitchRowProps) {
  const id = useId();
  return (
    <label className="flex cursor-pointer items-start justify-between gap-4">
      <span className="grid gap-0.5">
        <span id={`${id}-label`} className="text-sm font-medium">
          {label}
        </span>
        <span id={`${id}-description`} className="text-xs text-muted-foreground">
          {description}
        </span>
      </span>
      <Switch
        className="mt-0.5"
        aria-labelledby={`${id}-label`}
        aria-describedby={`${id}-description`}
        checked={checked}
        onCheckedChange={onCheckedChange}
      />
    </label>
  );
}
