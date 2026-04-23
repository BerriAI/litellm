import React from "react";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import type { ExportScope, EntityType } from "./types";

interface ExportTypeSelectorProps {
  value: ExportScope;
  onChange: (value: ExportScope) => void;
  entityType: EntityType;
}

const items: Array<{
  value: ExportScope;
  title: (t: EntityType) => string;
  description: (t: EntityType) => string;
}> = [
  {
    value: "daily",
    title: (t) => `Day-by-day breakdown by ${t}`,
    description: (t) => `Daily metrics for each ${t}`,
  },
  {
    value: "daily_with_keys",
    title: (t) => `Day-by-day breakdown by ${t} and key`,
    description: (t) => `Daily metrics for each ${t}, split by API key`,
  },
  {
    value: "daily_with_models",
    title: (t) => `Day-by-day by ${t} and model`,
    description: () => `Daily metrics split by model`,
  },
];

const ExportTypeSelector: React.FC<ExportTypeSelectorProps> = ({
  value,
  onChange,
  entityType,
}) => {
  return (
    <div>
      <Label className="text-sm font-medium block mb-2">Export type</Label>
      <RadioGroup
        value={value}
        onValueChange={(v) => onChange(v as ExportScope)}
        className="w-full space-y-2"
      >
        {items.map((item) => (
          <label
            key={item.value}
            className="flex items-start p-3 border border-border rounded-lg hover:bg-muted cursor-pointer transition-colors"
          >
            <RadioGroupItem value={item.value} className="mt-0.5" />
            <div className="ml-3 flex-1">
              <div className="font-medium text-sm">
                {item.title(entityType)}
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {item.description(entityType)}
              </div>
            </div>
          </label>
        ))}
      </RadioGroup>
    </div>
  );
};

export default ExportTypeSelector;
