import React from "react";
import { useTranslation } from "react-i18next";

import { type ComplexityRouterConfigValue, usesLlmClassifier } from "./ComplexityRouterConfig";
import { restrictedBy } from "./TierRestrictions";

const TierConfigIntro: React.FC<{ value: ComplexityRouterConfigValue }> = ({ value }) => {
  const { t } = useTranslation();
  return (
    <div className="mb-4 space-y-2 text-sm text-muted-foreground">
      <p>{t("addModel.tierConfigIntro.chooseModels")}</p>
      <p className="text-xs">
        {restrictedBy(value, "displayNames", t)?.reason ?? t("addModel.tierConfigIntro.displayNamesAppear")}
        {!value.custom_tier_set &&
          usesLlmClassifier(value.classifier_type) &&
          t("addModel.tierConfigIntro.judgeUsesNames")}
      </p>
    </div>
  );
};

export default TierConfigIntro;
