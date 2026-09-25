import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  STREAM_SCOPE_OPTIONS,
  type GuardrailStreamScope,
  isGuardrailStreamScope,
} from "./guardrail_info_helpers";

const STREAM_SCOPE_ITEMS = STREAM_SCOPE_OPTIONS.map((option) => ({
  label: option.label,
  value: option.value,
}));

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
