import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { GuardrailField, labelWithHint, type GuardrailFormControl } from "./GuardrailFormField";
import {
  STREAM_SCOPE_OPTIONS,
  formatGuardrailStreamScope,
  type GuardrailStreamScope,
  isGuardrailStreamScope,
} from "./guardrail_info_helpers";

const STREAM_SCOPE_ITEMS = STREAM_SCOPE_OPTIONS.map((option) => ({
  label: option.label,
  value: option.value,
}));

const REQUEST_SHAPE_HINT =
  "Run this guardrail on streaming requests, non-streaming requests, or both, for each selected mode.";

export const StreamScopeFields = ({
  modes,
  value,
  onChange,
}: {
  modes: string[];
  value: Record<string, GuardrailStreamScope>;
  onChange: (next: Record<string, GuardrailStreamScope>) => void;
}) => {
  if (modes.length === 0) return null;

  return (
    <div className="space-y-3">
      {modes.map((mode) => {
        const selected = value[mode] ?? "both";
        return (
          <div key={mode} className="space-y-1">
            <label htmlFor={`stream-scope-${mode}`} className="block text-sm font-medium text-foreground">
              {mode} applies to
            </label>
            <Select
              items={STREAM_SCOPE_ITEMS}
              value={selected}
              onValueChange={(next: GuardrailStreamScope | null) => {
                if (!isGuardrailStreamScope(next)) return;
                onChange({ ...value, [mode]: next });
              }}
            >
              <SelectTrigger id={`stream-scope-${mode}`} className="w-full">
                <SelectValue placeholder="Streaming and non-streaming" />
              </SelectTrigger>
              <SelectContent>
                {STREAM_SCOPE_OPTIONS.map((option) => (
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

export const StreamScopeFormField = ({ control, modes }: { control: GuardrailFormControl; modes: string[] }) => (
  <GuardrailField
    control={control}
    name="stream_scope_by_mode"
    label={labelWithHint("Request shape", REQUEST_SHAPE_HINT)}
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

export const GuardrailStreamScopeCaption = ({ raw }: { raw: unknown }) => {
  const label = formatGuardrailStreamScope(raw);
  if (!label) return null;
  return <p className="mt-1 text-sm text-muted-foreground">{label}</p>;
};

export const GuardrailStreamScopeDetail = ({ raw }: { raw: unknown }) => {
  const label = formatGuardrailStreamScope(raw);
  if (!label) return null;
  return (
    <div>
      <p className="font-medium">Request shape</p>
      <div>{label}</div>
    </div>
  );
};
