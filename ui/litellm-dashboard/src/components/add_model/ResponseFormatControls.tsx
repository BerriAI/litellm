import { useLanguage } from "@/contexts/LanguageContext";
import { Switch } from "@/components/ui/switch";
import React from "react";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const ResponseFormatControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => {
  const { t } = useLanguage();
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={value.return_raw_model_name ?? false}
          onCheckedChange={(returnRawModelName) => onChange({ ...value, return_raw_model_name: returnRawModelName })}
          aria-label={t("Return raw model name")}
        />
        <strong className="font-semibold">{t("Return raw model name")}</strong>
      </div>
      <span className="block text-xs text-muted-foreground">
        {t("Return the resolved underlying model name in responses instead of the autorouter alias.")}
      </span>
      <div className="mt-4 flex items-center gap-2 mb-2">
        <Switch
          checked={value.max_tokens_from_tier_model ?? true}
          onCheckedChange={(enabled) => onChange({ ...value, max_tokens_from_tier_model: enabled })}
          aria-label={t("Cap max_tokens at the tier model's output ceiling")}
        />
        <strong className="font-semibold">{t("Cap max_tokens at the tier model's output ceiling")}</strong>
      </div>
      <span className="block text-xs text-muted-foreground">
        {t(
          "Replace the caller's max_tokens with the routed tier model's output ceiling so one client value fits every tier. Off forwards the caller's value unchanged.",
        )}
      </span>
    </>
  );
};

export default ResponseFormatControls;
