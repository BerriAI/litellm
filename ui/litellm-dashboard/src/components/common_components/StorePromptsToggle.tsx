import { useId } from "react";
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import { Switch } from "@/components/ui/switch";

interface StorePromptsToggleProps {
  value?: unknown;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}

export function StorePromptsToggle({ value, onChange, disabled }: StorePromptsToggleProps) {
  const id = useId();
  return (
    <Field>
      <FieldLabel htmlFor={id}>Store prompts in spend logs</FieldLabel>
      <Switch id={id} checked={value !== false} onCheckedChange={onChange} disabled={disabled} />
      <FieldDescription>
        Turn off to exclude prompts and responses from spend logs. Global and team restrictions still apply.
      </FieldDescription>
    </Field>
  );
}

export function StorePromptsMetadataToggle({
  value,
  onChange,
}: {
  value: string | undefined;
  onChange: (value: string) => void;
}) {
  const metadata = readMetadata(value);
  return (
    <StorePromptsToggle
      value={metadata?.store_prompts_in_spend_logs}
      disabled={metadata === null}
      onChange={(store) => onChange(JSON.stringify({ ...metadata, store_prompts_in_spend_logs: store }, null, 2))}
    />
  );
}

function readMetadata(value: string | undefined): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(value || "{}");
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}
