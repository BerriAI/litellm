import React from "react";
import { useTranslation } from "react-i18next";
import { Switch } from "@/components/ui/switch";
import { MultiSelect } from "@/components/shared/MultiSelect";
import TierRowSelect from "./TierRowSelect";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const PlanModeOverrideControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  planModeTierOptions: { value: string; label: string }[];
}> = ({ value, onChange, planModeTierOptions }) => {
  const { t } = useTranslation();
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={value.plan_mode_min_tier !== undefined}
          disabled={planModeTierOptions.length === 0}
          onCheckedChange={(enabled) =>
            onChange({
              ...value,
              plan_mode_min_tier: enabled ? planModeTierOptions.at(-1)?.value : undefined,
            })
          }
          aria-label={t("addModel.planModeOverrideControls.routeToMinimumTier")}
        />
        <strong className="font-semibold">{t("addModel.planModeOverrideControls.routeToMinimumTier")}</strong>
      </div>
      <span className="block text-xs mb-3 text-muted-foreground">
        {t("addModel.planModeOverrideControls.description")}
        {planModeTierOptions.length === 0 && t("addModel.planModeOverrideControls.addModelsToEnable")}
      </span>
      {value.plan_mode_min_tier !== undefined && (
        <div style={{ maxWidth: 320 }}>
          <TierRowSelect
            label={t("addModel.planModeOverrideControls.minimumTier")}
            options={planModeTierOptions}
            value={value.plan_mode_min_tier ?? null}
            onValueChange={(tier) => onChange({ ...value, plan_mode_min_tier: tier })}
          />
        </div>
      )}
      <div className="mt-4">
        <strong className="mb-1 block font-semibold">
          {t("addModel.planModeOverrideControls.additionalSentinels")}
        </strong>
        <MultiSelect
          options={(value.plan_mode_patterns ?? []).map((pattern) => ({ label: pattern, value: pattern }))}
          value={value.plan_mode_patterns ?? []}
          onValueChange={(patterns) =>
            onChange({ ...value, plan_mode_patterns: patterns.length > 0 ? patterns : undefined })
          }
          placeholder={t("addModel.planModeOverrideControls.sentinelPlaceholder")}
          emptyText={t("addModel.planModeOverrideControls.typeToAddSentinel")}
          allowCustomValues
          className="w-full"
        />
        <span className="mt-1 block text-xs text-muted-foreground">
          {t("addModel.planModeOverrideControls.sentinelsHelp")}
        </span>
      </div>
    </>
  );
};

export default PlanModeOverrideControls;
