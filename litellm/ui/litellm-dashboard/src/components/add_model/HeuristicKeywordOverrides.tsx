import React from "react";
import { useTranslation } from "react-i18next";
import { MultiSelect } from "@/components/shared/MultiSelect";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const fields = [
  ["code_keywords", "addModel.heuristicKeywordOverrides.fields.code"],
  ["reasoning_keywords", "addModel.heuristicKeywordOverrides.fields.reasoning"],
  ["technical_keywords", "addModel.heuristicKeywordOverrides.fields.technical"],
  ["simple_keywords", "addModel.heuristicKeywordOverrides.fields.simple"],
] as const;

const HeuristicKeywordOverrides: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => {
  const { t } = useTranslation();
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">{t("addModel.heuristicKeywordOverrides.description")}</p>
      {fields.map(([key, label]) => {
        const keywords = value[key] ?? [];
        return (
          <div key={key}>
            <strong className="mb-1 block font-semibold">{t(label)}</strong>
            <MultiSelect
              options={keywords.map((keyword) => ({ label: keyword, value: keyword }))}
              value={keywords}
              onValueChange={(next) => onChange({ ...value, [key]: next.length > 0 ? next : undefined })}
              placeholder={t("addModel.heuristicKeywordOverrides.addPlaceholder", { label: t(label).toLowerCase() })}
              emptyText={t("addModel.heuristicKeywordOverrides.typeToAddKeyword")}
              allowCustomValues
              className="w-full"
            />
          </div>
        );
      })}
    </div>
  );
};

export default HeuristicKeywordOverrides;
