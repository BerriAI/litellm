import React, { useId } from "react";
import { AutoRouterAllowanceNote } from "./AutoRouterAvailability";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import ClassifierCircuitBreakerConfig from "./ClassifierCircuitBreakerConfig";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { defaultJevClassifierConfig } from "./jev_classifier_config";

export default function JevClassifierConfig({
  value,
  onChange,
}: {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}) {
  const id = useId();
  const config = value.jev_classifier_config ?? defaultJevClassifierConfig();
  const provider = config.provider ?? "typesafe";
  const modelLabel = provider === "laya" ? "Laya" : "Jev";
  const update = (patch: Partial<typeof config>) =>
    onChange({ ...value, jev_classifier_config: { ...config, ...patch } });

  return (
    <div className="mt-4 space-y-3">
      <p className="text-sm text-muted-foreground">Uses a decision model to choose between your configured tiers</p>
      <div>
        <Label htmlFor={`${id}-provider`}>Decision model</Label>
        <Select
          value={provider}
          items={[
            { value: "typesafe", label: "Jev (TypeSafe)" },
            { value: "laya", label: "Laya (self-hosted)" },
          ]}
          onValueChange={(next) => {
            if ((next === "typesafe" || next === "laya") && next !== provider) {
              update({ provider: next, model: next === "laya" ? "multilingual" : "jev-latest" });
            }
          }}
        >
          <SelectTrigger id={`${id}-provider`} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="typesafe">Jev (TypeSafe)</SelectItem>
            <SelectItem value="laya">Laya (self-hosted)</SelectItem>
          </SelectContent>
        </Select>
        <p className="mt-1 text-xs text-muted-foreground">
          {provider === "laya"
            ? "Uses the self-hosted Laya deployment configured by your gateway administrator"
            : "Uses the TypeSafe connection configured by your gateway administrator"}
        </p>
      </div>
      <div>
        <Label htmlFor={`${id}-model`}>{modelLabel} Model</Label>
        <Input id={`${id}-model`} value={config.model} onChange={(event) => update({ model: event.target.value })} />
        {provider === "laya" && (
          <p className="mt-1 text-xs text-muted-foreground">Choose multilingual, english, or typed-decisions</p>
        )}
      </div>
      <div>
        <Label htmlFor={`${id}-timeout`}>{modelLabel} Timeout (ms)</Label>
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
        <Label htmlFor={`${id}-instructions`}>{modelLabel} Instructions</Label>
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
            Restore built-in {modelLabel} instructions
          </Button>
        )}
        <p className="text-xs text-muted-foreground">
          Built-in {modelLabel} is available without a license and uses the shipped tier criteria
        </p>
      </div>
    </div>
  );
}
