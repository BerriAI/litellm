import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import React from "react";
import { useTranslation } from "react-i18next";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const OUTPUT_TOKEN_POLICIES = [
  {
    value: true,
    label: "addModel.responseFormatControls.outputTokenPolicies.modelCeiling.label",
    description: "addModel.responseFormatControls.outputTokenPolicies.modelCeiling.description",
  },
  {
    value: "clamp",
    label: "addModel.responseFormatControls.outputTokenPolicies.callerBudget.label",
    description: "addModel.responseFormatControls.outputTokenPolicies.callerBudget.description",
  },
  {
    value: false,
    label: "addModel.responseFormatControls.outputTokenPolicies.passThrough.label",
    description: "addModel.responseFormatControls.outputTokenPolicies.passThrough.description",
  },
] as const;

const ResponseFormatControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => {
  const { t } = useTranslation();
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={value.return_raw_model_name ?? false}
          onCheckedChange={(returnRawModelName) => onChange({ ...value, return_raw_model_name: returnRawModelName })}
          aria-label={t("addModel.responseFormatControls.rawModelName.label")}
        />
        <strong className="font-semibold">{t("addModel.responseFormatControls.rawModelName.label")}</strong>
      </div>
      <span className="block text-xs text-muted-foreground">
        {t("addModel.responseFormatControls.rawModelName.help")}
      </span>
      <div className="mt-4 space-y-2">
        <strong className="font-semibold">{t("addModel.responseFormatControls.outputTokenLimit.label")}</strong>
        <RadioGroup
          value={value.max_tokens_from_tier_model ?? true}
          onValueChange={(mode: unknown) => {
            if (typeof mode === "boolean" || mode === "clamp") {
              onChange({ ...value, max_tokens_from_tier_model: mode });
            }
          }}
          aria-label={t("addModel.responseFormatControls.outputTokenLimit.label")}
        >
          {OUTPUT_TOKEN_POLICIES.map((policy) => (
            <Label key={String(policy.value)} className="items-start font-normal leading-normal">
              <RadioGroupItem value={policy.value} className="mt-0.5" />
              <span>
                <span className="block font-medium">{t(policy.label)}</span>
                <span className="block text-xs text-muted-foreground">{t(policy.description)}</span>
              </span>
            </Label>
          ))}
        </RadioGroup>
        <span className="block text-xs text-muted-foreground">
          {t("addModel.responseFormatControls.outputTokenLimit.note")}
        </span>
      </div>
    </>
  );
};

export default ResponseFormatControls;
