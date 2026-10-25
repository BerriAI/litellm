import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";

import { Switch } from "@/components/ui/switch";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

export const ModalityRoutingControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => {
  const { t } = useLanguage();
  const modalityRouting = value.modality_routing ?? false;
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={modalityRouting}
          onCheckedChange={(nextModalityRouting) => onChange({ ...value, modality_routing: nextModalityRouting })}
          aria-label={t("Route image requests to vision-capable models")}
        />
        <strong className="font-semibold">{t("Route image requests to vision-capable models")}</strong>
      </div>
      <span className="block text-xs mb-3 text-muted-foreground">
        {t(
          "Replaces a routed model that cannot take image input with the nearest higher tier that can, then the default model, instead of failing with a provider 400. Only models explicitly declared supports_vision false are replaced, and a kept session pin still wins unless you turn on the override below.",
        )}
      </span>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={value.modality_pin_override ?? false}
          onCheckedChange={(modalityPinOverride) => onChange({ ...value, modality_pin_override: modalityPinOverride })}
          disabled={!modalityRouting}
          aria-label={t("Override session pin for image requests")}
        />
        <strong className="font-semibold">{t("Override session pin for image requests")}</strong>
      </div>
      <span className="block text-xs text-muted-foreground">
        {t(
          "Route an image turn to a capable model even when the session is pinned to one that cannot take images. The pin is kept, so the next text turn goes back to it. Needs image routing turned on.",
        )}
      </span>
    </>
  );
};
