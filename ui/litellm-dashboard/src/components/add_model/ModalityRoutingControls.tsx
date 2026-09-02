import React from "react";

import { Switch } from "@/components/ui/switch";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

export const ModalityRoutingControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => (
  <>
    <div className="flex items-center gap-2 mb-2">
      <Switch
        checked={value.modality_routing ?? false}
        onCheckedChange={(modalityRouting) => onChange({ ...value, modality_routing: modalityRouting })}
        aria-label="Route image requests to vision-capable models"
      />
      <strong className="font-semibold">Route image requests to vision-capable models</strong>
    </div>
    <span className="block text-xs text-muted-foreground">
      Replaces a routed model that cannot take image input with the nearest higher tier that can, then the default
      model, instead of failing with a provider 400. Only models explicitly declared supports_vision false are replaced,
      and a kept session pin still wins.
    </span>
  </>
);
