import React from "react";
import { useTranslation } from "react-i18next";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import type { ExportType, EntityType } from "./types";

interface ExportTypeSelectorProps {
  value: ExportType;
  onChange: (value: ExportType) => void;
  entityType: EntityType;
}

const ExportTypeSelector: React.FC<ExportTypeSelectorProps> = ({ value, onChange, entityType }) => {
  const { t } = useTranslation();
  const allScopes: { value: ExportType; title: string; description: string }[] = [
    {
      value: "daily",
      title: t("usage.export.type.daily.title", { entityType }),
      description: t("usage.export.type.daily.description", { entityType }),
    },
    {
      value: "daily_with_keys",
      title: t("usage.export.type.dailyWithKeys.title", { entityType }),
      description: t("usage.export.type.dailyWithKeys.description", { entityType }),
    },
    {
      value: "daily_with_models",
      title: t("usage.export.type.dailyWithModels.title", { entityType }),
      description: t("usage.export.type.dailyWithModels.description"),
    },
    {
      value: "daily_with_users",
      title: t("usage.export.type.dailyWithUsers.title", { entityType }),
      description: t("usage.export.type.dailyWithUsers.description", { entityType }),
    },
  ];
  const scopes = allScopes.filter((scope) => scope.value !== "daily_with_users" || entityType !== "user");

  return (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">{t("usage.export.type.label")}</label>
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
