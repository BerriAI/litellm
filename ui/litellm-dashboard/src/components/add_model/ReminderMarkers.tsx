import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { getReminderMarkersError, type ReminderMarkerPair } from "./build_complexity_router_config";

const ReminderMarkers: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  showValidationErrors?: boolean;
}> = ({ value, onChange, showValidationErrors = false }) => {
  const { t } = useTranslation();
  const markers = value.reminder_markers ?? [];
  const update = (index: number, patch: Partial<ReminderMarkerPair>) =>
    onChange({
      ...value,
      reminder_markers: markers.map((marker, markerIndex) =>
        markerIndex === index ? { ...marker, ...patch } : marker,
      ),
    });
  const remove = (index: number) => {
    const next = markers.filter((_, markerIndex) => markerIndex !== index);
    onChange({ ...value, reminder_markers: next.length > 0 ? next : undefined });
  };
  const error = getReminderMarkersError(value.reminder_markers, t);
  return (
    <div>
      <p className="mb-4 text-sm text-muted-foreground">
        <Trans
          i18nKey="addModel.reminderMarkers.description"
          components={{ marker: <code>&lt;system-reminder&gt;</code> }}
        />
      </p>
      <div className="space-y-3">
        {markers.map((marker, index) => (
          <div className="flex items-end gap-2" key={index}>
            <div className="flex-1">
              <label className="mb-1 block text-sm font-medium" htmlFor={`reminder-marker-${index}-open`}>
                {t("addModel.reminderMarkers.openingTag")}
              </label>
              <Input
                id={`reminder-marker-${index}-open`}
                placeholder="<note>"
                value={marker.open}
                onChange={(event) => update(index, { open: event.target.value })}
              />
            </div>
            <div className="flex-1">
              <label className="mb-1 block text-sm font-medium" htmlFor={`reminder-marker-${index}-close`}>
                {t("addModel.reminderMarkers.closingTag")}
              </label>
              <Input
                id={`reminder-marker-${index}-close`}
                placeholder="</note>"
                value={marker.close}
                onChange={(event) => update(index, { close: event.target.value })}
              />
            </div>
            <Button
              variant="ghost"
              size="icon"
              aria-label={t("addModel.reminderMarkers.removePairAria", { number: index + 1 })}
              onClick={() => remove(index)}
            >
              <Trash2 />
            </Button>
          </div>
        ))}
      </div>
      <Button
        className="mt-3"
        variant="outline"
        onClick={() => onChange({ ...value, reminder_markers: [...markers, { open: "", close: "" }] })}
      >
        <Plus />
        {t("addModel.reminderMarkers.addPair")}
      </Button>
      {showValidationErrors && error && <p className="mt-2 text-xs text-destructive">{error}</p>}
    </div>
  );
};

export default ReminderMarkers;
