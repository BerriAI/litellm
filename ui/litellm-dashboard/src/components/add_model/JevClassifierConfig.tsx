import React, { useId } from "react";
import { AutoRouterAllowanceNote } from "./AutoRouterAvailability";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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
  const isLaya = config.provider === "laya";
  const name = isLaya ? "Laya" : "Jev";
  const update = (patch: Partial<typeof config>) =>
    onChange({ ...value, jev_classifier_config: { ...config, ...patch } });

  return (
    <div className="mt-4 space-y-3">
      <p className="text-sm text-muted-foreground">
        {isLaya
          ? "Uses your self-hosted Laya server to choose from your configured tiers"
          : "Uses TypeSafe System One Choice evaluation with your configured tiers"}
      </p>
      {isLaya && (
        <>
          <div>
            <Label htmlFor={`${id}-base`}>Laya Server URL</Label>
            <Input
              id={`${id}-base`}
              type="url"
              placeholder="http://localhost:8000"
              value={config.api_base ?? ""}
              onChange={(event) => update({ api_base: event.target.value })}
            />
            <p className="text-xs text-muted-foreground">Base URL reachable from the gateway, without /v1/systemone</p>
          </div>
          <div>
            <Label htmlFor={`${id}-key`}>Laya API Key (optional)</Label>
            <Input
              id={`${id}-key`}
              type="password"
              autoComplete="new-password"
              value={config.api_key ?? ""}
              placeholder="Leave blank unless your Laya server requires a key"
              onChange={(event) => update({ api_key: event.target.value })}
            />
          </div>
        </>
      )}
      <div>
        <Label htmlFor={`${id}-model`}>{name} Model</Label>
        <Input id={`${id}-model`} value={config.model} onChange={(event) => update({ model: event.target.value })} />
      </div>
      <div>
        <Label htmlFor={`${id}-timeout`}>{name} Timeout (ms)</Label>
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
        <Label htmlFor={`${id}-instructions`}>{name} Instructions</Label>
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
            Restore built-in {name} instructions
          </Button>
        )}
        <p className="text-xs text-muted-foreground">
          Built-in {name} is available without a license and uses the shipped tier criteria
        </p>
      </div>
    </div>
  );
}
