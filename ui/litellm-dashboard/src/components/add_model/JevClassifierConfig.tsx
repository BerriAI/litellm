import React, { useId } from "react";
import { AutoRouterAllowanceNote } from "./AutoRouterAvailability";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import ClassifierCircuitBreakerConfig from "./ClassifierCircuitBreakerConfig";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { defaultJevClassifierConfig, transitionDecisionModelProvider } from "./jev_classifier_config";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

export default function JevClassifierConfig({
  value,
  onChange,
}: {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}) {
  const id = useId();
  const { userRole, isViewOnly } = useAuthorized();
  const config = value.jev_classifier_config ?? defaultJevClassifierConfig();
  const provider = config.provider ?? "typesafe";
  const providerLabel = provider === "bespoke_nimble" ? "Nimble" : "Jev";
  const update = (patch: Partial<typeof config>) =>
    onChange({ ...value, jev_classifier_config: { ...config, ...patch } });

  return (
    <div className="mt-4 space-y-3">
      <fieldset className="space-y-2">
        <legend className="text-sm font-medium">Decision model provider</legend>
        <RadioGroup
          value={provider}
          onValueChange={(next) => {
            if (next === "typesafe" || next === "bespoke_nimble") {
              onChange({ ...value, jev_classifier_config: transitionDecisionModelProvider(config, next) });
            }
          }}
          className="flex flex-wrap gap-4"
        >
          <Label>
            <RadioGroupItem value="typesafe" />
            TypeSafe Jev
          </Label>
          <Label>
            <RadioGroupItem value="bespoke_nimble" />
            Nimble (open source)
          </Label>
        </RadioGroup>
      </fieldset>
      <p className="text-sm text-muted-foreground">
        {providerLabel} selects a tier for each request using your configured tier criteria
      </p>
      <div>
        <Label htmlFor={`${id}-model`}>{providerLabel} Model</Label>
        <Input id={`${id}-model`} value={config.model} onChange={(event) => update({ model: event.target.value })} />
      </div>
      <div>
        <Label htmlFor={`${id}-timeout`}>{providerLabel} Timeout (ms)</Label>
        <Input
          id={`${id}-timeout`}
          type="number"
          min={1}
          step={1}
          value={config.timeout_ms}
          onChange={(event) => update({ timeout_ms: Number(event.target.value) })}
        />
      </div>
      {isProxyAdminRole(userRole ?? "") && !isViewOnly && (
        <details className="space-y-3 rounded-lg border p-3">
          <summary className="cursor-pointer text-sm font-medium">Connection settings</summary>
          <p className="text-xs text-muted-foreground">
            Saved connection values are hidden. Untouched fields keep the saved connection. New routers use the gateway
            connection when these fields are blank
          </p>
          <div>
            <Label htmlFor={`${id}-api-base`}>API Base</Label>
            <Input
              id={`${id}-api-base`}
              value={config.api_base ?? ""}
              placeholder={provider === "bespoke_nimble" ? "https://nimble.example.com" : "https://api.typesafe.ai"}
              onChange={(event) => update({ api_base: event.target.value })}
            />
          </div>
          <div>
            <Label htmlFor={`${id}-api-key`}>API Key</Label>
            <Input
              id={`${id}-api-key`}
              type="password"
              autoComplete="new-password"
              value={config.api_key ?? ""}
              placeholder={
                provider === "bespoke_nimble" ? "Optional for a keyless Nimble server" : "Enter the endpoint's key"
              }
              onChange={(event) => update({ api_key: event.target.value || null })}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" type="button" onClick={() => update({ api_base: null, api_key: null })}>
              Use gateway connection
            </Button>
            {provider === "bespoke_nimble" && (
              <>
                <Button variant="outline" type="button" onClick={() => update({ api_key: null })}>
                  Clear saved API key
                </Button>
                <p className="w-full text-xs text-muted-foreground">
                  Clearing the key keeps a saved custom endpoint and connects without authentication. A gateway
                  connection still uses its configured key. Use gateway connection clears both router overrides
                </p>
              </>
            )}
          </div>
          {config.api_base === null && config.api_key === null && (
            <p role="status" className="text-xs text-muted-foreground">
              Gateway connection selected for the next save
            </p>
          )}
        </details>
      )}
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
        <Label htmlFor={`${id}-instructions`}>{providerLabel} Instructions</Label>
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
            Restore built-in {providerLabel} instructions
          </Button>
        )}
        <p className="text-xs text-muted-foreground">
          Built-in decision model classification is available without a license and uses the shipped tier criteria
        </p>
      </div>
    </div>
  );
}
