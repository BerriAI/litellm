import React, { useId } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { AutoRouterAllowanceNote } from "./AutoRouterAvailability";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import ClassifierCircuitBreakerConfig from "./ClassifierCircuitBreakerConfig";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { defaultJevClassifierConfig, fixedClassifierModels, type OssClassifierProvider } from "./jev_classifier_config";

const providerDescriptions: Record<OssClassifierProvider, ParseKeys> = {
  jev: "addModel.jevClassifierConfig.providerDescriptions.jev",
  laya: "addModel.jevClassifierConfig.providerDescriptions.laya",
  bespoke: "addModel.jevClassifierConfig.providerDescriptions.bespoke",
  databricks: "addModel.jevClassifierConfig.providerDescriptions.databricks",
};

export default function JevClassifierConfig({
  value,
  onChange,
  section = "all",
}: {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  section?: "all" | "model" | "advanced";
}) {
  const { t } = useTranslation();
  const id = useId();
  const config = value.jev_classifier_config ?? defaultJevClassifierConfig();
  const models = fixedClassifierModels(config.provider);
  const update = (patch: Partial<typeof config>) =>
    onChange({ ...value, jev_classifier_config: { ...config, ...patch } });

  return (
    <div className={section === "model" ? "space-y-3" : "mt-4 space-y-3"}>
      {section !== "advanced" && !config.deployment_name && (
        <>
          <p className="text-sm text-muted-foreground">{t(providerDescriptions[config.provider ?? "jev"])}</p>
          <div className="space-y-2">
            <Label htmlFor={`${id}-model`}>{t("addModel.jevClassifierConfig.model.label")}</Label>
            {models ? (
              <Select value={config.model} onValueChange={(model) => model && update({ model })}>
                <SelectTrigger id={`${id}-model`} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input
                id={`${id}-model`}
                value={config.model}
                placeholder={
                  config.provider === "databricks"
                    ? t("addModel.jevClassifierConfig.model.databricksPlaceholder")
                    : undefined
                }
                onChange={(event) => update({ model: event.target.value })}
              />
            )}
          </div>
        </>
      )}
      {section !== "model" && (
        <>
          <div className="space-y-2">
            <Label htmlFor={`${id}-timeout`}>{t("addModel.jevClassifierConfig.timeout")}</Label>
            <Input
              id={`${id}-timeout`}
              type="number"
              min={1}
              step={1}
              value={config.timeout_ms}
              onChange={(event) => update({ timeout_ms: Number(event.target.value) })}
            />
          </div>
          <ClassifierCircuitBreakerConfig
            value={config}
            onChange={(next) =>
              update({
                circuit_breaker_enabled: next.circuit_breaker_enabled,
                circuit_breaker_cooldown_seconds: next.circuit_breaker_cooldown_seconds,
              })
            }
          />
          <div className="space-y-2">
            <Label htmlFor={`${id}-instructions`}>{t("addModel.jevClassifierConfig.instructions.label")}</Label>
            <AutoRouterAllowanceNote
              feature="tier_or_classifier_prompt"
              label={t("addModel.jevClassifierConfig.instructions.allowance")}
            />
            <Textarea
              id={`${id}-instructions`}
              value={config.instructions ?? ""}
              placeholder={t("addModel.jevClassifierConfig.instructions.placeholder")}
              onChange={(event) => update({ instructions: event.target.value || undefined })}
            />
            {config.instructions && (
              <Button variant="outline" type="button" onClick={() => update({ instructions: undefined })}>
                {t("addModel.jevClassifierConfig.instructions.restore")}
              </Button>
            )}
            <p className="text-xs text-muted-foreground">{t("addModel.jevClassifierConfig.licenseNote")}</p>
          </div>
        </>
      )}
    </div>
  );
}
