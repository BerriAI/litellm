import { useAccessGroups } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { useModelAccessGroups } from "@/app/(dashboard)/hooks/modelAccessGroups/useModelAccessGroups";
import { modelsWithinAccess } from "./modelThrottling";
import { useId } from "react";
import { Plus, X } from "lucide-react";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Switch } from "@/components/ui/switch";
import type { ModelThrottling, ModelThrottlingThreshold } from "./modelThrottling";

interface Props {
  value: ModelThrottling | null;
  onChange: (value: ModelThrottling | null) => void;
  maxBudget: unknown;
  availableModels: string[];
  disabled?: boolean;
}

export function ModelThrottlingEditor({ value, onChange, maxBudget, availableModels, disabled = false }: Props) {
  const id = useId();
  const limit = Number(maxBudget);
  const hasBudget = Number.isFinite(limit) && limit > 0;
  const rows = value?.thresholds ?? [];
  const updateRow = (index: number, update: Partial<ModelThrottlingThreshold>) =>
    onChange({ thresholds: rows.map((row, i) => (i === index ? { ...row, ...update } : row)) });

  return (
    <Field className="my-4">
      <div className="flex items-center justify-between gap-4">
        <FieldLabel htmlFor={`${id}-enabled`}>Model throttling</FieldLabel>
        <Switch
          id={`${id}-enabled`}
          checked={value !== null}
          disabled={disabled || (!hasBudget && value === null)}
          onCheckedChange={(enabled) => onChange(enabled ? { thresholds: [{ spend: 0, models: [] }] } : null)}
        />
      </div>
      <FieldDescription>
        {hasBudget
          ? "Make your budget go further by pausing higher-cost models as spending increases."
          : "Set a final budget to enable model throttling."}
      </FieldDescription>
      {value !== null && (
        <div className="space-y-3">
          {rows.map((row, index) => {
            const amountInvalid = row.spend > 0 && (!hasBudget || row.spend >= limit);
            const usedModels = rows.filter((_, i) => i !== index).flatMap((other) => other.models);
            return (
              <div
                key={index}
                className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 rounded-lg border border-border p-3 sm:grid-cols-[140px_minmax(0,1fr)_auto]"
              >
                <Field className="col-span-2 sm:col-span-1">
                  <FieldLabel htmlFor={`${id}-spend-${index}`}>Spend reaches</FieldLabel>
                  <InputGroup>
                    <InputGroupAddon>
                      <InputGroupText>$</InputGroupText>
                    </InputGroupAddon>
                    <InputGroupInput
                      id={`${id}-spend-${index}`}
                      type="number"
                      min={0}
                      step="any"
                      value={row.spend || ""}
                      placeholder="0.00"
                      disabled={disabled}
                      aria-invalid={amountInvalid}
                      aria-describedby={amountInvalid ? `${id}-error-${index}` : undefined}
                      onChange={(event) =>
                        updateRow(index, {
                          spend: Number.isNaN(event.target.valueAsNumber) ? 0 : event.target.valueAsNumber,
                        })
                      }
                    />
                  </InputGroup>
                  {amountInvalid && (
                    <p id={`${id}-error-${index}`} className="text-sm text-destructive">
                      Must be below the final budget
                    </p>
                  )}
                </Field>
                <Field>
                  <FieldLabel htmlFor={`${id}-models-${index}`}>Pause models</FieldLabel>
                  <MultiSelect
                    id={`${id}-models-${index}`}
                    options={availableModels
                      .filter((model) => !usedModels.includes(model))
                      .map((model) => ({ label: model, value: model }))}
                    value={row.models}
                    onValueChange={(models) => updateRow(index, { models })}
                    placeholder="Select models"
                    emptyText="No other available models"
                    disabled={disabled}
                  />
                </Field>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="self-end"
                  disabled={disabled}
                  aria-label={`Remove threshold ${index + 1}`}
                  onClick={() =>
                    onChange(rows.length === 1 ? null : { thresholds: rows.filter((_, i) => i !== index) })
                  }
                >
                  <X className="size-4" />
                </Button>
              </div>
            );
          })}
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={disabled}
            onClick={() => onChange({ thresholds: [...rows, { spend: 0, models: [] }] })}
          >
            <Plus className="size-4" />
            Add threshold
          </Button>
          <FieldDescription>Access returns when spend resets. The final budget limit still applies.</FieldDescription>
        </div>
      )}
    </Field>
  );
}

export function ModelThrottlingField({
  allowedModels,
  accessGroupIds = [],
  ...props
}: Props & { allowedModels?: string[] | null; accessGroupIds?: string[] }) {
  const { data: modelGroups = [] } = useModelAccessGroups();
  const { data: accessGroups = [] } = useAccessGroups();
  const groups = Object.fromEntries(modelGroups.map((group) => [group.access_group, group.model_names]));
  const grants = accessGroups
    .filter((group) => accessGroupIds.includes(group.access_group_id))
    .flatMap((group) => group.access_model_names);
  return (
    <ModelThrottlingEditor
      {...props}
      availableModels={modelsWithinAccess(props.availableModels, allowedModels, groups, grants)}
    />
  );
}
