import React from "react";
import { $api } from "@/lib/http/api";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { fuseProfileFields, selectFuseProfile, type FuseSettings } from "./forecast_classifier_config";

const catalogQueryOptions = {
  staleTime: Infinity,
  gcTime: Infinity,
  retry: false,
  refetchOnMount: false,
  refetchOnWindowFocus: false,
  refetchOnReconnect: false,
};

export default function FuseProfilePresets({
  value,
  onChange,
}: {
  value: FuseSettings;
  onChange: (value: FuseSettings) => void;
}) {
  const id = React.useId();
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
        independent of routing model names
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
      {fuseProfileFields.map((field) => {
        const label = {
          efficient_profile: "Efficient solver profile",
          capable_profile: "Capable solver profile",
          harness: "Harness and budget",
        }[field];
        const presetId = value[`${field}_preset`];
        const presets = field === "harness" ? data?.harnesses : data?.models;
        const preset = presets?.find((entry) => entry.id === presetId);
        const custom = value[field] != null || presetId == null;
        const effectiveText = value[field] ?? preset?.text ?? "";
        return (
          <div key={field} className="space-y-2 min-w-0">
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
              onValueChange={(selected) => {
                if (selected)
                  onChange(
                    selectFuseProfile(value, field, selected === "custom" ? undefined : selected, effectiveText),
                  );
              }}
            />
            <Textarea
              aria-label={label}
              value={effectiveText}
              readOnly={!custom}
              maxLength={4000}
              rows={4}
              placeholder={
                field === "harness"
                  ? "Tools, execution environment, verification, and budget available to each solver"
                  : "Describe this solver's strengths, limitations, and settings"
              }
              onChange={(event) => onChange({ ...value, [field]: event.target.value })}
            />
            {presetId && (
              <div className="space-y-1 text-xs text-muted-foreground break-words">
                <p>
                  {custom ? "Custom text overrides preset" : "Preset"}: {presetId}
                </p>
                {preset ? (
                  <>
                    <p>Catalog version: {data?.version}</p>
                    {"model" in preset && typeof preset.model === "string" && <p>Model: {preset.model}</p>}
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
      })}
    </div>
  );
}
