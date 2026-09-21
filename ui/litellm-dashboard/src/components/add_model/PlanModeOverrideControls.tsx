import React from "react";
import { Switch } from "@/components/ui/switch";
import TierRowSelect from "./TierRowSelect";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const PlanModeOverrideControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  planModeTierOptions: { value: string; label: string }[];
}> = ({ value, onChange, planModeTierOptions }) => (
  <>
    <div className="flex items-center gap-2 mb-2">
      <Switch
        checked={value.plan_mode_min_tier !== undefined}
        disabled={planModeTierOptions.length === 0}
        onCheckedChange={(enabled) =>
          onChange({
            ...value,
            plan_mode_min_tier: enabled ? planModeTierOptions.at(-1)?.value : undefined,
          })
        }
        aria-label="Route plan-mode requests to a minimum tier"
      />
      <strong className="font-semibold">Route plan-mode requests to a minimum tier</strong>
    </div>
    <span className="block text-xs mb-3 text-muted-foreground">
      Requests from coding agents in plan mode (Claude Code, GitHub Copilot) route to at least this tier. The classifier
      still wins when it picks higher, and the override only lasts while plan mode is active.
      {planModeTierOptions.length === 0 && " Add models to a tier to enable this."}
    </span>
    {value.plan_mode_min_tier !== undefined && (
      <div style={{ maxWidth: 320 }}>
        <TierRowSelect
          label="Plan-mode minimum tier"
          options={planModeTierOptions}
          value={value.plan_mode_min_tier ?? null}
          onValueChange={(tier) => onChange({ ...value, plan_mode_min_tier: tier })}
        />
      </div>
    )}
  </>
);

export default PlanModeOverrideControls;
