import React from "react";
import i18n from "@/i18n/i18n";
import type { ParseKeys, TFunction } from "i18next";
import { useTranslation } from "react-i18next";
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

const profileLabels: Readonly<Record<FuseProfileField, ParseKeys>> = {
  efficient_profile: "addModel.fuseProfilePresets.profiles.efficient",
  capable_profile: "addModel.fuseProfilePresets.profiles.capable",
  harness: "addModel.fuseProfilePresets.profiles.harness",
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

const profileSelectionLabel = (
  awaitingCustomText: boolean,
  custom: boolean,
  t: TFunction = i18n.t.bind(i18n),
): string => {
  if (awaitingCustomText) return t("addModel.fuseProfilePresets.selection.awaitingCustomText");
  if (custom) return t("addModel.fuseProfilePresets.selection.customOverridesPreset");
  return t("addModel.fuseProfilePresets.selection.preset");
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
  const { t } = useTranslation();
  const label = t(profileLabels[field]);
  const presetId = value[`${field}_preset`];
  const preset = presets?.find((entry) => entry.id === presetId);
  const awaitingCustomText = customWithoutPreview.has(field) && value[field] == null && presetId != null;
  const custom = value[field] != null || presetId == null || awaitingCustomText;
  const effectiveText = value[field] ?? preset?.text ?? "";
  const selectionLabel = profileSelectionLabel(awaitingCustomText, custom, t);
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
      ? t("addModel.fuseProfilePresets.placeholders.harness")
      : t("addModel.fuseProfilePresets.placeholders.solver");

  return (
    <div className="space-y-2 min-w-0">
      <Label htmlFor={`${id}-${field}-preset`}>{t("addModel.fuseProfilePresets.presetLabel", { label })}</Label>
      <SearchSelect
        inputId={`${id}-${field}-preset`}
        aria-label={t("addModel.fuseProfilePresets.presetLabel", { label })}
        value={custom ? "custom" : presetId}
        allowClear={false}
        options={[
          { value: "custom", label: t("addModel.fuseProfilePresets.custom") },
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
          aria-label={t("addModel.fuseProfilePresets.keepSavedAriaLabel", { label: label.toLowerCase() })}
          onClick={keepSavedPreset}
        >
          {t("addModel.fuseProfilePresets.keepSaved")}
        </Button>
      )}
      {presetId && (
        <div className="space-y-1 text-xs text-muted-foreground break-words">
          <p>
            {selectionLabel}: {presetId}
          </p>
          {preset ? (
            <>
              <p>{t("addModel.fuseProfilePresets.catalogVersion", { version: catalogVersion })}</p>
              {preset.model && <p>{t("addModel.fuseProfilePresets.model", { model: preset.model })}</p>}
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                {preset.sources.map((source, index) => (
                  <a key={source} href={source} target="_blank" rel="noopener noreferrer" className="underline">
                    {t("addModel.fuseProfilePresets.source", { number: index + 1 })}
                  </a>
                ))}
              </div>
            </>
          ) : (
            <p>{t("addModel.fuseProfilePresets.previewUnavailable")}</p>
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
  const { t } = useTranslation();
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
      <p className="text-xs text-muted-foreground">{t("addModel.fuseProfilePresets.intro")}</p>
      {isPending && (
        <p role="status" className="text-xs text-muted-foreground">
          {t("addModel.fuseProfilePresets.loading")}
        </p>
      )}
      {isError && (
        <p role="status" className="text-xs text-muted-foreground">
          {t("addModel.fuseProfilePresets.loadError")}
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
