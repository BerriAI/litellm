import React from "react";
import i18n from "@/i18n/i18n";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { Info } from "lucide-react";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { SimpleTooltip } from "@/components/ui/tooltip";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { isForecastClassifier } from "./forecast_classifier_config";
import { resolveComplexityDefaultModel } from "./tier_rows";

interface DefaultModelFieldProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelOptions: { value: string; label: string }[];
}

const defaultModelPlaceholderFor = (
  derivedDefaultModel: string | undefined,
  isCustomSet: boolean,
  t: TFunction = i18n.t.bind(i18n),
): string => {
  if (derivedDefaultModel) return t("addModel.defaultModelField.placeholders.derived", { model: derivedDefaultModel });
  return isCustomSet
    ? t("addModel.defaultModelField.placeholders.customSet")
    : t("addModel.defaultModelField.placeholders.builtIn");
};

const DefaultModelField = ({ value, onChange, modelOptions }: DefaultModelFieldProps) => {
  const { t } = useTranslation();
  const defaultModelPlaceholder = defaultModelPlaceholderFor(
    resolveComplexityDefaultModel(value),
    Boolean(value.custom_tier_set),
    t,
  );
  // Clearing the select drops the key entirely rather than storing "", so an emptied pin reads as
  // "track the tiers" everywhere downstream instead of as a blank model name.
  const handleDefaultModelChange = (model: string | null | undefined) => {
    onChange({ ...value, default_model: model || undefined });
  };

  return (
    <div className="mt-4 mb-2" role="group" aria-label={t("addModel.defaultModelField.groupAriaLabel")}>
      <div className="flex items-center gap-2 mb-2">
        <strong className="text-base font-semibold">{t("addModel.defaultModelField.title")}</strong>
        <SimpleTooltip content={t("addModel.defaultModelField.tooltip")}>
          <Info className="size-4 text-muted-foreground" />
        </SimpleTooltip>
      </div>
      <SearchSelect
        options={modelOptions}
        value={value.default_model ?? ""}
        onValueChange={handleDefaultModelChange}
        placeholder={defaultModelPlaceholder}
        emptyText={t("addModel.defaultModelField.empty")}
        aria-label={t("addModel.defaultModelField.ariaLabel")}
      />
      <span className="block mt-1 text-xs text-muted-foreground">
        {isForecastClassifier(value.classifier_type)
          ? t("addModel.defaultModelField.help.forecast")
          : t("addModel.defaultModelField.help.standard")}
      </span>
    </div>
  );
};

export default DefaultModelField;
