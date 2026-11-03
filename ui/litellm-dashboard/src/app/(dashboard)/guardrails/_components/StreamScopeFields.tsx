import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { GuardrailField, labelWithHint, type GuardrailFormControl } from "./GuardrailFormField";
import {
  STREAM_SCOPE_OPTIONS,
  formatGuardrailStreamScope,
  type GuardrailStreamScope,
  isGuardrailStreamScope,
} from "./guardrail_info_helpers";

export const StreamScopeFields = ({
  modes,
  value,
  onChange,
}: {
  modes: string[];
  value: Record<string, GuardrailStreamScope>;
  onChange: (next: Record<string, GuardrailStreamScope>) => void;
}) => {
  const { t } = useTranslation();
  if (modes.length === 0) return null;

  const streamScopeItems = STREAM_SCOPE_OPTIONS.map((option) => ({
    label: t(option.labelKey),
    value: option.value,
  }));

  return (
    <div className="space-y-3">
      {modes.map((mode) => {
        const selected = value[mode] ?? "both";
        return (
          <div key={mode} className="space-y-1">
            <label htmlFor={`stream-scope-${mode}`} className="block text-sm font-medium text-foreground">
              {t("guardrails.streamScope.appliesTo", { mode })}
            </label>
            <Select
              items={streamScopeItems}
              value={selected}
              onValueChange={(next: GuardrailStreamScope | null) => {
                if (!isGuardrailStreamScope(next)) return;
                onChange({ ...value, [mode]: next });
              }}
            >
              <SelectTrigger id={`stream-scope-${mode}`} className="w-full">
                <SelectValue placeholder={t("guardrails.streamScope.placeholder")} />
              </SelectTrigger>
              <SelectContent>
                {streamScopeItems.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        );
      })}
    </div>
  );
};

export const StreamScopeFormField = ({ control, modes }: { control: GuardrailFormControl; modes: string[] }) => {
  const { t } = useTranslation();
  return (
    <GuardrailField
      control={control}
      name="stream_scope_by_mode"
      label={labelWithHint(t("guardrails.streamScope.requestShape"), t("guardrails.streamScope.requestShapeHint"))}
    >
      {({ value, onChange }) => (
        <StreamScopeFields
          modes={modes}
          value={(value as Record<string, GuardrailStreamScope> | undefined) ?? {}}
          onChange={onChange}
        />
      )}
    </GuardrailField>
  );
};

export const GuardrailStreamScopeCaption = ({ raw }: { raw: unknown }) => {
  const { t } = useTranslation();
  const label = formatGuardrailStreamScope(raw, t);
  if (!label) return null;
  return <p className="mt-1 text-sm text-muted-foreground">{label}</p>;
};

export const GuardrailStreamScopeDetail = ({ raw }: { raw: unknown }) => {
  const { t } = useTranslation();
  const label = formatGuardrailStreamScope(raw, t);
  if (!label) return null;
  return (
    <div>
      <p className="font-medium">{t("guardrails.streamScope.requestShape")}</p>
      <div>{label}</div>
    </div>
  );
};
