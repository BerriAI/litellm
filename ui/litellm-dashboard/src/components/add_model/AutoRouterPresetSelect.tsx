import React from "react";
import { Select, SelectContent, SelectItem, SelectSeparator, SelectTrigger, SelectValue } from "@/components/ui/select";
import { type AutoRouterPreset } from "@/lib/autorouter_presets";

export type PresetAvailability =
  | { kind: "available"; viaDeployments: boolean }
  | { kind: "loading" }
  | { kind: "unverifiable" }
  | { kind: "missing_models"; models: readonly string[] };

// Every non-"available" state disables the option. Selection derives from this same function
// (see presetAvailability below), so an option a caller can click is always one that can be applied.
const presetDisabledHint = (availability: PresetAvailability): string | null => {
  switch (availability.kind) {
    case "available":
      return null;
    case "loading":
      return "Checking model availability...";
    case "unverifiable":
      return "Cannot verify these models are available";
    case "missing_models":
      return `Missing: ${availability.models.join(", ")}`;
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
      { value: "custom", label: "Start from scratch" },
    ],
    [sortedPresetOptions],
  );

  return (
    <div className="mb-4 w-full space-y-2">
      <div className="space-y-2">
        <Select
          items={templateItems}
          value={selectedPreset ?? null}
          onValueChange={(presetKey: string | null) => onPresetChange(presetKey ?? undefined)}
        >
          <SelectTrigger aria-label="Use a template" data-testid="template-selector" className="w-full">
            <SelectValue placeholder="Use a template" />
          </SelectTrigger>
          <SelectContent className="[&_[data-slot=select-item]>div]:min-w-0 [&_[data-slot=select-item]>div]:shrink [&_[data-slot=select-item]>div]:whitespace-normal [&_[data-slot=select-item]>div]:break-words">
            {sortedPresetOptions.map(({ preset, availability: presetState }) => {
              const disabledHint = presetDisabledHint(presetState);
              const hintClass = isPresetHintAlarming(presetState) ? "text-destructive" : "text-muted-foreground";
              const matchedHint =
                presetState.kind === "available" && presetState.viaDeployments ? "Matches your deployments" : null;

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
                    <div className="text-xs text-muted-foreground">Replaces classifier and model choices</div>
                    {disabledHint && <div className={`text-xs mt-1 ${hintClass}`}>{disabledHint}</div>}
                    {matchedHint && <div className="text-xs mt-1 text-success">{matchedHint}</div>}
                  </div>
                </SelectItem>
              );
            })}
            <SelectSeparator className="mx-0" />
            <SelectItem value="custom" label="Start from scratch">
              <div>
                <div className="font-medium">Start from scratch</div>
                <div className="text-xs text-muted-foreground">
                  Clears tier models, switches to Heuristics, and resets routing settings
                </div>
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">Replaces classifier and model choices</p>
        {pending && <div className="text-xs mt-1 text-muted-foreground">Loading templates...</div>}
        {unavailable && (
          <div className="text-xs mt-1 text-destructive">
            Could not load templates, so only Start from scratch is shown.{" "}
            <button type="button" className="underline" onClick={onRetry}>
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
