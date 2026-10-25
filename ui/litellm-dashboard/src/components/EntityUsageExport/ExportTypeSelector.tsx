import React from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import type { ExportType, EntityType } from "./types";
import { useLanguage } from "@/contexts/LanguageContext";

interface ExportTypeSelectorProps {
  value: ExportType;
  onChange: (value: ExportType) => void;
  entityType: EntityType;
}

const ExportTypeSelector: React.FC<ExportTypeSelectorProps> = ({ value, onChange, entityType }) => {
  const { t } = useLanguage();
  const entity = t(entityType);
  const allScopes: { value: ExportType; title: string; description: string }[] = [
    {
      value: "daily",
      title: t("Day-by-day breakdown by {entity}", { entity }),
      description: t("Daily metrics for each {entity}", { entity }),
    },
    {
      value: "daily_with_keys",
      title: t("Day-by-day breakdown by {entity} and key", { entity }),
      description: t("Daily metrics for each {entity}, split by API key", { entity }),
    },
    {
      value: "daily_with_models",
      title: t("Day-by-day by {entity} and model", { entity }),
      description: t("Daily metrics split by model"),
    },
    {
      value: "daily_with_users",
      title: t("Day-by-day breakdown by {entity} and user", { entity }),
      description: t("Daily metrics for each {entity}, split by key owner", { entity }),
    },
  ];
  const scopes = allScopes.filter((scope) => scope.value !== "daily_with_users" || entityType !== "user");

  return (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">{t("Export type")}</label>
      <RadioGroup value={value} onValueChange={(next) => onChange(next as ExportType)} className="gap-2">
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
