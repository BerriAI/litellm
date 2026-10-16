import React from "react";
import { $api } from "@/lib/http/api";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  fuseProfileFields,
  selectFuseProfile,
  type FuseProfileField,
  type FuseSettings,
} from "./forecast_classifier_config";

const catalogQueryOptions = {
  staleTime: Infinity,
  gcTime: Infinity,
  retry: false,
  refetchOnMount: false,
  refetchOnWindowFocus: false,
  refetchOnReconnect: false,
};

const profileLabels: Readonly<Record<FuseProfileField, string>> = {
  efficient_profile: "Efficient solver profile",
  capable_profile: "Capable solver profile",
  harness: "Harness and budget",
};

type FusePresetEntry = {
  id: string;
  label: string;
  text: string;
  sources: readonly string[];
  model?: string;
};

type FieldProps = {
  id: string;
  field: FuseProfileField;
  value: FuseSettings;
  onChange: (value: FuseSettings) => void;
  presets: readonly FusePresetEntry[] | undefined;
  catalogVersion: string | undefined;
  customWithoutPreview: ReadonlySet<FuseProfileField>;
  setCustomWithoutPreview: React.Dispatch<React.SetStateAction<ReadonlySet<FuseProfileField>>>;
};

const profileSelectionLabel = (awaitingCustomText: boolean, custom: boolean): string => {
  if (awaitingCustomText) return "Saved preset remains active until replacement text is entered";
  if (custom) return "Custom text overrides preset";
  return "Preset";
};

function FuseProfilePresetField({
  id,
  field,
  value,
  onChange,
  presets,
  catalogVersion,
  customWithoutPreview,
  setCustomWithoutPreview,
}: FieldProps) {
  const label = profileLabels[field];
  const presetId = value[`${field}_preset`];
  const preset = presets?.find((entry) => entry.id === presetId);
  const awaitingCustomText = customWithoutPreview.has(field) && value[field] == null && presetId != null;
  const custom = value[field] != null || presetId == null || awaitingCustomText;
  const effectiveText = value[field] ?? preset?.text ?? "";
  const selectionLabel = profileSelectionLabel(awaitingCustomText, custom);
  const chooseProfile = (selected: string | null) => {
    if (!selected) return;
    const missingPreview = presetId != null && preset == null && value[field] == null;
    if (selected === "custom" && missingPreview) {
      setCustomWithoutPreview((fields) => new Set([...fields, field]));
      return;
    }
    setCustomWithoutPreview((fields) => new Set([...fields].filter((entry) => entry !== field)));
    onChange(selectFuseProfile(value, field, selected === "custom" ? undefined : selected, effectiveText));
  };
  const editProfile = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (customWithoutPreview.has(field) && event.target.value.trim().length > 0) {
      setCustomWithoutPreview((fields) => new Set([...fields].filter((entry) => entry !== field)));
      onChange(selectFuseProfile(value, field, undefined, event.target.value));
      return;
    }
    onChange({ ...value, [field]: event.target.value });
  };
  const keepSavedPreset = () => {
    if (presetId == null) return;
    setCustomWithoutPreview((fields) => new Set([...fields].filter((entry) => entry !== field)));
    onChange(selectFuseProfile(value, field, presetId, ""));
  };
  const placeholder =
    field === "harness"
      ? "Tools, execution environment, verification, and budget available to each solver"
      : "Describe this solver's strengths, limitations, and settings";

  return (
    <div className="space-y-2 min-w-0">
      <Label htmlFor={`${id}-${field}-preset`}>{label} preset</Label>
      <SearchSelect
        inputId={`${id}-${field}-preset`}
        aria-label={`${label} preset`}
        value={custom ? "custom" : presetId}
        allowClear={false}
        options={[
          { value: "custom", label: "Custom" },
          ...(presets ?? []).map((entry) => ({ value: entry.id, label: entry.label, sublabel: entry.id })),
        ]}
        onValueChange={chooseProfile}
      />
      <Textarea
        aria-label={label}
        value={effectiveText}
        readOnly={!custom}
        maxLength={4000}
        rows={4}
        placeholder={placeholder}
        onChange={editProfile}
      />
      {customWithoutPreview.has(field) && presetId != null && (
        <Button
          type="button"
          variant="outline"
          size="sm"
          aria-label={`Keep saved ${label.toLowerCase()} preset`}
          onClick={keepSavedPreset}
        >
          Keep saved preset
        </Button>
      )}
      {presetId && (
        <div className="space-y-1 text-xs text-muted-foreground break-words">
          <p>
            {selectionLabel}: {presetId}
          </p>
          {preset ? (
            <>
              <p>Catalog version: {catalogVersion}</p>
              {preset.model && <p>Model: {preset.model}</p>}
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                {preset.sources.map((source, index) => (
                  <a key={source} href={source} target="_blank" rel="noopener noreferrer" className="underline">
                    Source {index + 1}
                  </a>
                ))}
              </div>
            </>
          ) : (
            <p>Preset preview unavailable. The saved reference is preserved</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function FuseProfilePresets({
  value,
  onChange,
}: {
  value: FuseSettings;
  onChange: (value: FuseSettings) => void;
}) {
  const id = React.useId();
  const [customWithoutPreview, setCustomWithoutPreview] = React.useState<ReadonlySet<FuseProfileField>>(
    () => new Set(),
  );
  const { data, isPending, isError } = $api.useQuery(
    "get",
    "/public/complexity_router/fuse_presets",
    {},
    catalogQueryOptions,
  );
  return (
    <div className="space-y-4 min-w-0">
      <p className="text-xs text-muted-foreground">
        Choose profiles that match every deployment in each solver group and its actual settings. Profile selection is
        independent of routing model names. Presets describe the solvers and runtime; they do not set a quality gap or
        calibration. Validate those separately for your workload, judge, and exact profile versions.
      </p>
      {isPending && (
        <p role="status" className="text-xs text-muted-foreground">
          Loading profile presets. Custom editing is available
        </p>
      )}
      {isError && (
        <p role="status" className="text-xs text-muted-foreground">
          Profile presets could not be loaded. Saved references are preserved and Custom editing is available
        </p>
      )}
      {fuseProfileFields.map((field) => (
        <FuseProfilePresetField
          key={field}
          id={id}
          field={field}
          value={value}
          onChange={onChange}
          presets={field === "harness" ? data?.harnesses : data?.models}
          catalogVersion={data?.version}
          customWithoutPreview={customWithoutPreview}
          setCustomWithoutPreview={setCustomWithoutPreview}
        />
      ))}
    </div>
  );
}
