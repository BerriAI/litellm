import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import React from "react";
import { useTranslation } from "react-i18next";

import type { ClassifierLLMConfig } from "./ComplexityRouterConfig";

export const DEFAULT_CLASSIFIER_CIRCUIT_BREAKER_ENABLED = true;
export const DEFAULT_CLASSIFIER_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 30;

const COOLDOWN_ID = "classifier-circuit-breaker-cooldown-seconds";

interface ClassifierCircuitBreakerConfigProps {
  value: ClassifierLLMConfig;
  onChange: (value: ClassifierLLMConfig) => void;
}

const ClassifierCircuitBreakerConfig: React.FC<ClassifierCircuitBreakerConfigProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const [draftCooldown, setDraftCooldown] = React.useState<string | null>(null);
  const enabled = value.circuit_breaker_enabled ?? DEFAULT_CLASSIFIER_CIRCUIT_BREAKER_ENABLED;

  const handleCooldownChange = (raw: string) => {
    setDraftCooldown(raw);
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) return;
    onChange({
      ...value,
      circuit_breaker_cooldown_seconds: Math.max(1, Math.round(parsed)),
    });
  };

  return (
    <div className="space-y-2 rounded-md border border-border p-3">
      <div className="flex items-center gap-2">
        <Switch
          checked={enabled}
          onCheckedChange={(circuit_breaker_enabled) => onChange({ ...value, circuit_breaker_enabled })}
          aria-label={t("addModel.classifierCircuitBreakerConfig.label")}
        />
        <strong className="font-semibold">{t("addModel.classifierCircuitBreakerConfig.label")}</strong>
      </div>
      <span className="block text-xs text-muted-foreground">{t("addModel.classifierCircuitBreakerConfig.help")}</span>
      {enabled && (
        <div>
          <Label htmlFor={COOLDOWN_ID} className="block mb-1 font-semibold">
            {t("addModel.classifierCircuitBreakerConfig.cooldown")}
          </Label>
          <Input
            id={COOLDOWN_ID}
            type="text"
            inputMode="numeric"
            value={
              draftCooldown ??
              String(value.circuit_breaker_cooldown_seconds ?? DEFAULT_CLASSIFIER_CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            }
            onChange={(event) => handleCooldownChange(event.target.value)}
            onBlur={() => setDraftCooldown(null)}
            className="w-full"
          />
        </div>
      )}
    </div>
  );
};

export default ClassifierCircuitBreakerConfig;
