import React from "react";
import { useTranslation } from "react-i18next";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Switch } from "@/components/ui/switch";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const HousekeepingRoutingControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const enabled = value.route_housekeeping_to_cheapest_tier ?? true;
  const patterns = value.housekeeping_patterns ?? [];
  return (
    <>
      <div className="mb-2 flex items-center gap-2">
        <Switch
          checked={enabled}
          onCheckedChange={(next) => onChange({ ...value, route_housekeeping_to_cheapest_tier: next })}
          aria-label={t("addModel.housekeepingRoutingControls.routeToCheapest")}
        />
        <strong className="font-semibold">{t("addModel.housekeepingRoutingControls.routeToCheapest")}</strong>
      </div>
      <span className="mb-3 block text-xs text-muted-foreground">
        {t("addModel.housekeepingRoutingControls.description")}
      </span>
      <strong className="mb-1 block font-semibold">
        {t("addModel.housekeepingRoutingControls.additionalSentinels")}
      </strong>
      <MultiSelect
        options={patterns.map((pattern) => ({ label: pattern, value: pattern }))}
        value={patterns}
        onValueChange={(next) => onChange({ ...value, housekeeping_patterns: next.length > 0 ? next : undefined })}
        placeholder={t("addModel.housekeepingRoutingControls.sentinelPlaceholder")}
        emptyText={t("addModel.housekeepingRoutingControls.typeToAddSentinel")}
        allowCustomValues
        disabled={!enabled}
        className="w-full"
      />
      <span className="mt-1 block text-xs text-muted-foreground">
        {t("addModel.housekeepingRoutingControls.sentinelsHelp")}
        {!enabled && t("addModel.housekeepingRoutingControls.turnOnToTakeEffect")}
      </span>
    </>
  );
};

export default HousekeepingRoutingControls;
