import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Switch } from "@/components/ui/switch";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const HousekeepingRoutingControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => {
  const { t } = useLanguage();
  const enabled = value.route_housekeeping_to_cheapest_tier ?? true;
  const patterns = value.housekeeping_patterns ?? [];
  return (
    <>
      <div className="mb-2 flex items-center gap-2">
        <Switch
          checked={enabled}
          onCheckedChange={(next) => onChange({ ...value, route_housekeeping_to_cheapest_tier: next })}
          aria-label={t("Route housekeeping calls to the cheapest tier")}
        />
        <strong className="font-semibold">{t("Route housekeeping calls to the cheapest tier")}</strong>
      </div>
      <span className="mb-3 block text-xs text-muted-foreground">
        {t("Conversation-title style calls skip the classifier and go to the cheapest tier.")}
      </span>
      <strong className="mb-1 block font-semibold">{t("Additional housekeeping sentinels")}</strong>
      <MultiSelect
        options={patterns.map((pattern) => ({ label: pattern, value: pattern }))}
        value={patterns}
        onValueChange={(next) => onChange({ ...value, housekeeping_patterns: next.length > 0 ? next : undefined })}
        placeholder={t("e.g., conversation title")}
        emptyText={t("Type to add a sentinel")}
        allowCustomValues
        disabled={!enabled}
        className="w-full"
      />
      <span className="mt-1 block text-xs text-muted-foreground">
        {t("Case-sensitive literal strings added to the built-in conversation-title sentinels.")}
        {!enabled && t(" Turn housekeeping routing on for these to take effect.")}
      </span>
    </>
  );
};

export default HousekeepingRoutingControls;
