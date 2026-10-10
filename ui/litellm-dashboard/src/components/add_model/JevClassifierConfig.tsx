import React, { useId } from "react";
import { AutoRouterAllowanceNote } from "./AutoRouterAvailability";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import ClassifierCircuitBreakerConfig from "./ClassifierCircuitBreakerConfig";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { defaultJevClassifierConfig, fixedClassifierModels } from "./jev_classifier_config";

const providerDescriptions = {
  jev: "Uses TypeSafe System One Choice evaluation with your configured tiers",
  laya: "Uses Laya with your configured tiers. Set LAYA_API_BASE on the gateway to connect your Laya server.",
  bespoke:
    "Uses Bespoke Nimble with your configured tiers. Set BESPOKE_API_BASE on the gateway to connect your Nimble server.",
  databricks:
    "Uses a Databricks ai_decide serving endpoint with your configured tiers. Set DATABRICKS_API_BASE and DATABRICKS_API_KEY on the gateway, and enter the serving endpoint name as the classifier model.",
};

export default function JevClassifierConfig({
  value,
  onChange,
}: {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}) {
  const id = useId();
  const config = value.jev_classifier_config ?? defaultJevClassifierConfig();
  const models = fixedClassifierModels(config.provider);
  const update = (patch: Partial<typeof config>) =>
    onChange({ ...value, jev_classifier_config: { ...config, ...patch } });

  return (
    <div className="mt-4 space-y-3">
      <p className="text-sm text-muted-foreground">{providerDescriptions[config.provider ?? "jev"]}</p>
      <div>
        <Label htmlFor={`${id}-model`}>Classifier Model</Label>
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
              config.provider === "databricks" ? "Serving endpoint name, e.g. databricks-openjev-qwen35-4b" : undefined
            }
            onChange={(event) => update({ model: event.target.value })}
          />
        )}
      </div>
      <div>
        <Label htmlFor={`${id}-timeout`}>Classifier Timeout (ms)</Label>
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
      <div>
        <Label htmlFor={`${id}-instructions`}>Classifier Instructions</Label>
        <AutoRouterAllowanceNote
          feature="tier_or_classifier_prompt"
          label="Custom instructions share the custom-tier allowance"
        />
        <Textarea
          id={`${id}-instructions`}
          value={config.instructions ?? ""}
          placeholder="Leave blank to use the built-in instructions"
          onChange={(event) => update({ instructions: event.target.value || undefined })}
        />
        {config.instructions && (
          <Button variant="outline" type="button" onClick={() => update({ instructions: undefined })}>
            Restore built-in instructions
          </Button>
        )}
        <p className="text-xs text-muted-foreground">
          Built-in OSS classification is available without a license and uses the shipped tier criteria
        </p>
      </div>
    </div>
  );
}
