import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";

import { type ComplexityRouterConfigValue, usesLlmClassifier } from "./ComplexityRouterConfig";
import { restrictedBy } from "./TierRestrictions";

const TierConfigIntro: React.FC<{ value: ComplexityRouterConfigValue }> = ({ value }) => {
  const { t } = useLanguage();
  return (
    <div className="mb-4 space-y-2 text-sm text-muted-foreground">
      <p>{t("Choose one or more models for each tier. Requests use the models in their assigned tier")}</p>
      <p className="text-xs">
        {t(restrictedBy(value, "displayNames")?.reason ?? "Display names appear in the dashboard and spend logs")}
        {!value.custom_tier_set &&
          usesLlmClassifier(value.classifier_type) &&
          t(". Your judge model also uses these names to classify requests")}
      </p>
    </div>
  );
};

export default TierConfigIntro;
