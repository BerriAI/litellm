import React from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectSeparator, SelectTrigger, SelectValue } from "@/components/ui/select";
import { type AutoRouterPreset } from "@/lib/autorouter_presets";

export type PresetAvailability =
  | { kind: "available"; viaDeployments: boolean }
  | { kind: "loading" }
  | { kind: "unverifiable" }
  | { kind: "missing_models"; models: readonly string[] };

// Every non-"available" state disables the option. Selection derives from this same function
// (see presetAvailability below), so an option a caller can click is always one that can be applied.
const presetDisabledHint = (availability: PresetAvailability, t: TFunction): string | null => {
  switch (availability.kind) {
    case "available":
      return null;
    case "loading":
      return t("addModel.autoRouterPresetSelect.checkingAvailability");
    case "unverifiable":
      return t("addModel.autoRouterPresetSelect.cannotVerify");
    case "missing_models":
      return t("addModel.autoRouterPresetSelect.missingModels", { models: availability.models.join(", ") });
  }
};

// "loading"/"unverifiable" are transient system states, not a gap specific to this preset; only a
// caller-specific missing-model reason gets the alarming red treatment.
const isPresetHintAlarming = (availability: PresetAvailability): boolean => availability.kind === "missing_models";

type Props = {
  selectedPreset?: string;
  presets: AutoRouterPreset[];
  presetAvailability: (preset: AutoRouterPreset) => PresetAvailability;
  onPresetChange: (key: string | undefined) => void;
  pending: boolean;
  unavailable: boolean;
  onRetry: () => void;
};

export default function AutoRouterPresetSelect({
  selectedPreset,
  presets,
  presetAvailability,
  onPresetChange,
  pending,
  unavailable,
  onRetry,
}: Props) {
  const { t } = useTranslation();
  const sortedPresetOptions = React.useMemo(
    () =>
      presets
        .map((preset) => ({ preset, availability: presetAvailability(preset) }))
        .sort((a, b) => Number(b.availability.kind === "available") - Number(a.availability.kind === "available")),
    [presets, presetAvailability],
  );

  const templateItems = React.useMemo(
    () => [
      ...sortedPresetOptions.map(({ preset }) => ({ value: preset.key, label: preset.label })),
      { value: "custom", label: t("addModel.autoRouterPresetSelect.startFromScratch") },
    ],
    [sortedPresetOptions, t],
  );

  return (
    <div className="mb-4 w-full space-y-2">
      <div className="space-y-2">
        <Select
          items={templateItems}
          value={selectedPreset ?? null}
          onValueChange={(presetKey: string | null) => onPresetChange(presetKey ?? undefined)}
        >
          <SelectTrigger
            aria-label={t("addModel.autoRouterPresetSelect.useTemplate")}
            data-testid="template-selector"
            className="w-full"
          >
            <SelectValue placeholder={t("addModel.autoRouterPresetSelect.useTemplate")} />
          </SelectTrigger>
          <SelectContent className="[&_[data-slot=select-item]>div]:min-w-0 [&_[data-slot=select-item]>div]:shrink [&_[data-slot=select-item]>div]:whitespace-normal [&_[data-slot=select-item]>div]:break-words">
            {sortedPresetOptions.map(({ preset, availability: presetState }) => {
              const disabledHint = presetDisabledHint(presetState, t);
              const hintClass = isPresetHintAlarming(presetState) ? "text-destructive" : "text-muted-foreground";
              const matchedHint =
                presetState.kind === "available" && presetState.viaDeployments
                  ? t("addModel.autoRouterPresetSelect.matchesDeployments")
                  : null;

              return (
                <SelectItem
                  key={preset.key}
                  value={preset.key}
                  label={preset.label}
                  disabled={disabledHint !== null}
                  title={disabledHint ?? preset.description}
                >
                  <div>
                    <div className="font-medium">{preset.label}</div>
                    <div className="text-xs text-muted-foreground">{preset.description}</div>
                    <div className="text-xs text-muted-foreground">
                      {t("addModel.autoRouterPresetSelect.replacesChoices")}
                    </div>
                    {disabledHint && <div className={`text-xs mt-1 ${hintClass}`}>{disabledHint}</div>}
                    {matchedHint && <div className="text-xs mt-1 text-success">{matchedHint}</div>}
                  </div>
                </SelectItem>
              );
            })}
            <SelectSeparator className="mx-0" />
            <SelectItem value="custom" label={t("addModel.autoRouterPresetSelect.startFromScratch")}>
              <div>
                <div className="font-medium">{t("addModel.autoRouterPresetSelect.startFromScratch")}</div>
                <div className="text-xs text-muted-foreground">
                  {t("addModel.autoRouterPresetSelect.startFromScratchDescription")}
                </div>
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">{t("addModel.autoRouterPresetSelect.replacesChoices")}</p>
        {pending && (
          <div className="text-xs mt-1 text-muted-foreground">
            {t("addModel.autoRouterPresetSelect.loadingTemplates")}
          </div>
        )}
        {unavailable && (
          <div className="text-xs mt-1 text-destructive">
            {t("addModel.autoRouterPresetSelect.templatesUnavailable")}{" "}
            <button type="button" className="underline" onClick={onRetry}>
              {t("addModel.autoRouterPresetSelect.retry")}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
