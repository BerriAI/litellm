import React from "react";
import { useTranslation } from "react-i18next";

import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const NonReasoningTierToggle: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  available: boolean;
}> = ({ value, onChange, available }) => {
  const { t } = useTranslation();
  const handleToggle = (enabled: boolean): void => {
    const { NON_REASONING: existingPool, ...keptTiers } = value.tiers;
    // Turning it off must also release the plan-mode floor, which the backend rejects while it
    // names an inactive tier. An orphaned keyword rule is left for the save gate to name.
    const next: ComplexityRouterConfigValue = enabled
      ? { ...value, enable_non_reasoning_tier: true, tiers: { ...keptTiers, NON_REASONING: existingPool ?? [] } }
      : {
          ...value,
          enable_non_reasoning_tier: undefined,
          tiers: keptTiers,
          plan_mode_min_tier: value.plan_mode_min_tier === "NON_REASONING" ? undefined : value.plan_mode_min_tier,
        };
    onChange(next);
  };

  return (
    <>
      <Separator className="my-4" />
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={value.enable_non_reasoning_tier === true}
          disabled={!available}
          onCheckedChange={handleToggle}
          aria-label={t("addModel.nonReasoningTierToggle.addTier")}
        />
        <strong className="font-semibold">{t("addModel.nonReasoningTierToggle.addTier")}</strong>
      </div>
      <span className="block text-xs text-muted-foreground">
        {t("addModel.nonReasoningTierToggle.description")}
        {!available && t("addModel.nonReasoningTierToggle.requiresClassifier")}
      </span>
    </>
  );
};

export default NonReasoningTierToggle;
