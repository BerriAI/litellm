import React from "react";
import { useFormContext } from "react-hook-form";
import { FormField } from "@/components/shared/form/FormField";
import { PasswordInput } from "@/components/shared/PasswordInput";
import {
  Combobox,
  ComboboxClear,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { CacheField } from "./cacheSettingsFields";
import type { CacheFormValues } from "./cacheSettingsUtils";

export interface EmbeddingModelOption {
  value: string;
  label: string;
}

export const SECRET_ALREADY_SET_PLACEHOLDER = "Already set. Enter a new value to replace it.";

export const CONFIG_SOURCED_DESCRIPTION = "Set in config.yaml. Change it there, or remove it to edit this here.";

interface CacheFormFieldProps {
  field: CacheField;
  embeddingModels: EmbeddingModelOption[];
  isSecretConfigured?: boolean;
  isConfigSourced?: boolean;
}

const CacheFormField: React.FC<CacheFormFieldProps> = ({
  field,
  embeddingModels,
  isSecretConfigured = false,
  isConfigSourced = false,
}) => {
  const form = useFormContext<CacheFormValues>();
  const placeholder = isSecretConfigured ? SECRET_ALREADY_SET_PLACEHOLDER : field.helpText;
  const description = isConfigSourced ? CONFIG_SOURCED_DESCRIPTION : field.helpText;

  return (
    <FormField control={form.control} name={field.name} label={field.label} description={description}>
      {({ ref, value, onChange, ...rest }) => {
        if (field.type === "boolean") {
          return (
            <Switch
              {...rest}
              disabled={isConfigSourced}
              checked={value === true}
              onCheckedChange={(checked) => onChange(checked)}
            />
          );
        }
        if (field.type === "password") {
          return (
            <PasswordInput
              {...rest}
              ref={ref}
              disabled={isConfigSourced}
              value={typeof value === "string" ? value : ""}
              onChange={onChange}
              placeholder={placeholder}
              autoComplete="new-password"
            />
          );
        }
        if (field.type === "list") {
          return (
            <Textarea
              {...rest}
              ref={ref}
              disabled={isConfigSourced}
              rows={4}
              value={typeof value === "string" ? value : ""}
              onChange={onChange}
              placeholder={placeholder}
            />
          );
        }
        if (field.type === "model-select") {
          const selected = embeddingModels.find((model) => model.value === value) ?? null;
          return (
            <Combobox
              items={embeddingModels}
              value={selected}
              onValueChange={(model: EmbeddingModelOption | null) => onChange(model?.value ?? "")}
              itemToStringLabel={(model: EmbeddingModelOption) => model.label}
              isItemEqualToValue={(model: EmbeddingModelOption, other: EmbeddingModelOption) =>
                model.value === other.value
              }
            >
              <ComboboxInput
                {...rest}
                disabled={isConfigSourced}
                placeholder="Search and select a model..."
                className="w-full"
              >
                <ComboboxClear />
              </ComboboxInput>
              <ComboboxContent>
                <ComboboxEmpty>No models found</ComboboxEmpty>
                <ComboboxList>
                  {(model: EmbeddingModelOption) => (
                    <ComboboxItem key={model.value} value={model} title={model.label}>
                      {model.label}
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          );
        }
        return (
          <Input
            {...rest}
            ref={ref}
            disabled={isConfigSourced}
            inputMode={field.type === "integer" || field.type === "float" ? "decimal" : undefined}
            value={typeof value === "string" ? value : ""}
            onChange={onChange}
            placeholder={placeholder}
          />
        );
      }}
    </FormField>
  );
};

export default CacheFormField;
