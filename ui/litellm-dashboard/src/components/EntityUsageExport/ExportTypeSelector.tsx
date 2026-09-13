import React from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import type { ExportScope, EntityType } from "./types";

interface ExportTypeSelectorProps {
  value: ExportScope;
  onChange: (value: ExportScope) => void;
  entityType: EntityType;
}

const ExportTypeSelector: React.FC<ExportTypeSelectorProps> = ({ value, onChange, entityType }) => {
  const scopes: { value: ExportScope; title: string; description: string }[] = [
    {
      value: "daily",
      title: `Day-by-day breakdown by ${entityType}`,
      description: `Daily metrics for each ${entityType}`,
    },
    {
      value: "daily_with_keys",
      title: `Day-by-day breakdown by ${entityType} and key`,
      description: `Daily metrics for each ${entityType}, split by API key`,
    },
    {
      value: "daily_with_models",
      title: `Day-by-day by ${entityType} and model`,
      description: "Daily metrics split by model",
    },
  ];

  return (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">Export type</label>
      <RadioGroup value={value} onValueChange={(next) => onChange(next as ExportScope)} className="gap-2">
        {scopes.map((scope) => (
          <label
            key={scope.value}
            className="flex items-start p-3 border border-border rounded-lg hover:bg-accent cursor-pointer transition-colors"
          >
            <RadioGroupItem value={scope.value} className="mt-0.5" />
            <div className="ml-3 flex-1">
              <div className="font-medium text-sm">{scope.title}</div>
              <div className="text-xs text-muted-foreground mt-0.5">{scope.description}</div>
            </div>
          </label>
        ))}
      </RadioGroup>
    </div>
  );
};

export default ExportTypeSelector;
