import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { Info } from "lucide-react";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { SimpleTooltip } from "@/components/ui/tooltip";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { isForecastClassifier } from "./forecast_classifier_config";
import { resolveComplexityDefaultModel } from "./tier_rows";
import type { Translate } from "@/i18n/language";

interface DefaultModelFieldProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelOptions: { value: string; label: string }[];
}

const defaultModelPlaceholderFor = (
  derivedDefaultModel: string | undefined,
  isCustomSet: boolean,
  t: Translate,
): string => {
  if (derivedDefaultModel) return t("Derived from tiers: {model}", { model: derivedDefaultModel });
  return isCustomSet ? t("Add a model to your fallback tier") : t("Add a model to the Simple or Medium tier");
};

const DefaultModelField = ({ value, onChange, modelOptions }: DefaultModelFieldProps) => {
  const { t } = useLanguage();
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
    <div className="mt-4 mb-2" role="group" aria-label={t("Default model configuration")}>
      <div className="flex items-center gap-2 mb-2">
        <strong className="text-base font-semibold">{t("Default Model")}</strong>
        <SimpleTooltip
          content={t(
            "Leave empty to follow the tiers. A model chosen here is pinned: it stays the default however the tiers change.",
          )}
        >
          <Info className="size-4 text-muted-foreground" />
        </SimpleTooltip>
      </div>
      <SearchSelect
        options={modelOptions}
        value={value.default_model ?? ""}
        onValueChange={handleDefaultModelChange}
        placeholder={defaultModelPlaceholder}
        emptyText={t("No models found")}
        aria-label={t("Default model")}
      />
      <span className="block mt-1 text-xs text-muted-foreground">
        {isForecastClassifier(value.classifier_type)
          ? t("Used when routing cannot find a suitable model. Classifier failures route to the capable solver.")
          : t(
              'Used when the tier the request lands in has no model, and when the classifier fails with "Route to the default model" selected.',
            )}
      </span>
    </div>
  );
};

export default DefaultModelField;
