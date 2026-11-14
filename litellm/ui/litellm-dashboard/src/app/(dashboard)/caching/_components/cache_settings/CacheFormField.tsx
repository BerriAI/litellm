import React from "react";
import { useFormContext } from "react-hook-form";
import { useTranslation } from "react-i18next";
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { CacheField } from "./cacheSettingsFields";
import type { CacheFormValues } from "./cacheSettingsUtils";

export interface EmbeddingModelOption {
  value: string;
  label: string;
}

export const SECRET_ALREADY_SET_PLACEHOLDER = "caching.settings.field.secretAlreadySet";

interface CacheFormFieldProps {
  field: CacheField;
  embeddingModels: EmbeddingModelOption[];
  isSecretConfigured?: boolean;
}

const CacheFormField: React.FC<CacheFormFieldProps> = ({ field, embeddingModels, isSecretConfigured = false }) => {
  const { t } = useTranslation();
  const form = useFormContext<CacheFormValues>();
  const placeholder = isSecretConfigured ? t(SECRET_ALREADY_SET_PLACEHOLDER) : t(field.helpText);

  return (
    <FormField control={form.control} name={field.name} label={t(field.label)} description={t(field.helpText)}>
      {({ ref, value, onChange, ...rest }) => {
        if (field.type === "boolean") {
          return <Switch {...rest} checked={value === true} onCheckedChange={(checked) => onChange(checked)} />;
        }
        if (field.type === "password") {
          return (
            <PasswordInput
              {...rest}
              ref={ref}
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
              rows={4}
              value={typeof value === "string" ? value : ""}
              onChange={onChange}
              placeholder={placeholder}
            />
          );
        }
        if (field.type === "select") {
          const options = field.options ?? [];
          const { id, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy, name, onBlur, disabled } = rest;
          return (
            <Select
              items={options.map((option) => ({ label: t(option.label), value: option.value }))}
              name={name}
              disabled={disabled}
              value={typeof value === "string" && value !== "" ? value : null}
              onValueChange={onChange}
            >
              <SelectTrigger
                id={id}
                aria-invalid={ariaInvalid}
                aria-describedby={ariaDescribedBy}
                onBlur={onBlur}
                className="w-full"
              >
                <SelectValue placeholder={t("caching.settings.field.selectPlaceholder")} />
              </SelectTrigger>
              <SelectContent>
                {options.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {t(option.label)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          );
        }
        if (field.type === "model-select") {
          const selected = embeddingModels.find((model) => model.value === value) ?? null;
          return (
            <Combobox
              items={embeddingModels}
              value={selected}
              onValueChange={(model: EmbeddingModelOption | null) => onChange(model?.value ?? null)}
              itemToStringLabel={(model: EmbeddingModelOption) => model.label}
              isItemEqualToValue={(model: EmbeddingModelOption, other: EmbeddingModelOption) =>
                model.value === other.value
              }
            >
              <ComboboxInput
                {...rest}
                placeholder={t("caching.settings.field.modelSearchPlaceholder")}
                className="w-full"
              >
                <ComboboxClear />
              </ComboboxInput>
              <ComboboxContent>
                <ComboboxEmpty>{t("caching.settings.field.noModels")}</ComboboxEmpty>
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
