import React from "react";

import { Switch } from "@/components/ui/switch";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

/**
 * The opt-in fifth tier. Only offered on the LLM classification method, matching the backend: the
 * heuristic scorers cannot produce the tier, so enabling it there would buy a rubric bullet and a
 * model pool that no request ever reaches.
 */
const NonReasoningTierToggle: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  available: boolean;
}> = ({ value, onChange, available }) => {
  // Turning it off drops the tier's key rather than leaving an empty pool, which the backend
  // rejects; turning it back on restores whatever pool the form still held.
  const handleToggle = (enabled: boolean): void => {
    const { NON_REASONING: existingPool, ...keptTiers } = value.tiers;
    const next: ComplexityRouterConfigValue = {
      ...value,
      enable_non_reasoning_tier: enabled ? true : undefined,
      tiers: enabled ? { ...keptTiers, NON_REASONING: existingPool ?? [] } : keptTiers,
    };
    onChange(next);
  };

  return (
    <>
      <div className="flex items-center gap-2 mt-4 mb-2">
        <Switch
          checked={value.enable_non_reasoning_tier === true}
          disabled={!available}
          onCheckedChange={handleToggle}
          aria-label="Add a non-reasoning tier"
        />
        <strong className="font-semibold">Add a non-reasoning tier</strong>
      </div>
      <span className="block text-xs mb-3 text-muted-foreground">
        Adds NON_REASONING below Simple, for operational agent traffic that relays or reformats information rather than
        reasoning about it. Escalation still moves up out of it when a request needs more.
        {!available && " Requires the LLM classification method."}
      </span>
    </>
  );
};

export default NonReasoningTierToggle;
