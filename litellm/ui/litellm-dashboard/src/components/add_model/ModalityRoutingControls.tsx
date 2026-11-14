import React from "react";
import { useTranslation } from "react-i18next";

import { Switch } from "@/components/ui/switch";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

export const ModalityRoutingControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const modalityRouting = value.modality_routing ?? false;
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={modalityRouting}
          onCheckedChange={(nextModalityRouting) => onChange({ ...value, modality_routing: nextModalityRouting })}
          aria-label={t("addModel.modalityRoutingControls.routeImages")}
        />
        <strong className="font-semibold">{t("addModel.modalityRoutingControls.routeImages")}</strong>
      </div>
      <span className="block text-xs mb-3 text-muted-foreground">
        {t("addModel.modalityRoutingControls.routeImagesDescription")}
      </span>
      <div className="flex items-center gap-2 mb-2">
        <Switch
          checked={value.modality_pin_override ?? false}
          onCheckedChange={(modalityPinOverride) => onChange({ ...value, modality_pin_override: modalityPinOverride })}
          disabled={!modalityRouting}
          aria-label={t("addModel.modalityRoutingControls.overridePin")}
        />
        <strong className="font-semibold">{t("addModel.modalityRoutingControls.overridePin")}</strong>
      </div>
      <span className="block text-xs text-muted-foreground">
        {t("addModel.modalityRoutingControls.overridePinDescription")}
      </span>
    </>
  );
};
