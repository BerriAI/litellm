import React from "react";

import { Switch } from "@/components/ui/switch";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

export const HealthTierEscalationControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => (
  <>
    <div className="flex items-center gap-2 mb-2">
      <Switch
        checked={value.health_tier_escalation ?? false}
        onCheckedChange={(next) => onChange({ ...value, health_tier_escalation: next })}
        aria-label="Escalate to the next healthy tier when the selected tier is down"
      />
      <strong className="font-semibold">Escalate to the next healthy tier when the selected tier is down</strong>
    </div>
    <span className="block text-xs text-muted-foreground">
      When every model in the selected tier is in cooldown, send the request to the nearest higher tier that has a
      healthy model before using the default model. Never routes to a lower tier. Off by default: a dead tier then goes
      straight to the default model, or fails as before when none is set.
    </span>
  </>
);
