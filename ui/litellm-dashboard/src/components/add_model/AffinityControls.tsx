import React from "react";

import { Switch } from "@/components/ui/switch";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { DEFAULT_DEPLOYMENT_AFFINITY } from "./ComplexityRouterConfig";

export const AffinityControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => (
  <>
    <div className="flex items-center gap-2 mb-2">
      <Switch
        checked={value.deployment_affinity ?? DEFAULT_DEPLOYMENT_AFFINITY}
        onCheckedChange={(deploymentAffinity) => onChange({ ...value, deployment_affinity: deploymentAffinity })}
        aria-label="Pin a session to one deployment per model group"
      />
      <strong className="font-semibold">Pin a session to one deployment per model group</strong>
    </div>
    <span className="block text-xs text-muted-foreground">
      Keeps a session on the same deployment within a group, so provider prompt caches stay warm. Turn off to
      load-balance every turn.
    </span>
  </>
);
