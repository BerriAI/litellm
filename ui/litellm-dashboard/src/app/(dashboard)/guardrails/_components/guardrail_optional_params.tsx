import React from "react";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { PasswordInput } from "@/components/shared/PasswordInput";
import NumericalInput from "@/components/shared/numerical_input";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  asStringArray,
  asText,
  GuardrailField,
  readRecord,
  requiredRule,
  type GuardrailFieldControlProps,
  type GuardrailFormControl,
} from "./GuardrailFormField";

interface ProviderParam {
  param: string;
  description: string;
  required: boolean;
  default_value?: string;
  options?: string[];
  type?: string;
  fields?: { [key: string]: ProviderParam };
  dict_key_options?: string[];
  dict_value_type?: string;
}

interface GuardrailOptionalParamsProps {
  optionalParams: ProviderParam;
  parentFieldKey: string;
  control: GuardrailFormControl;
  values?: Record<string, unknown>;
}

interface DictFieldProps {
  field: ProviderParam;
  fullFieldKey: string;
  control: GuardrailFormControl;
  value: unknown;
}

const BOOLEAN_ITEMS = [
  { label: "True", value: true },
  { label: "False", value: false },
];

const isSecretKey = (fieldKey: string): boolean =>
  fieldKey.includes("password") || fieldKey.includes("secret") || fieldKey.includes("key");

const toNumberValue = (raw: unknown): unknown => {
  if (raw === null || raw === undefined || raw === "") return undefined;
  const num = Number(raw);
  return isNaN(num) ? raw : num;
};

const BooleanSelect: React.FC<{ control: GuardrailFieldControlProps; placeholder: string }> = ({
  control,
  placeholder,
}) => {
  const { id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy } = control;

  return (
    <Select
      items={BOOLEAN_ITEMS}
      value={typeof value === "boolean" ? value : null}
      onValueChange={(next: boolean | null) => onChange(next)}
    >
      <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy} className="w-full">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={true}>True</SelectItem>
        <SelectItem value={false}>False</SelectItem>
      </SelectContent>
    </Select>
  );
};

const DictField: React.FC<DictFieldProps> = ({ field, fullFieldKey, control, value }) => {
  const [selectedEntries, setSelectedEntries] = React.useState<Array<{ key: string; id: string }>>([]);
  const [availableKeys, setAvailableKeys] = React.useState<string[]>(field.dict_key_options || []);

  // Initialize selectedEntries and availableKeys based on existing value
  React.useEffect(() => {
    if (value && typeof value === "object") {
      const existingKeys = Object.keys(value);
      const entries = existingKeys.map((key) => ({
        key: key,
        id: `${key}_${Date.now()}_${Math.random()}`,
      }));
      setSelectedEntries(entries);

      const remainingKeys = (field.dict_key_options || []).filter((key) => !existingKeys.includes(key));
      setAvailableKeys(remainingKeys);
    }
  }, [value, field.dict_key_options]);

  const addEntry = (selectedKey: string) => {
    if (!selectedKey) return;

    const newEntry = {
      key: selectedKey,
      id: `${selectedKey}_${Date.now()}`,
    };

    setSelectedEntries([...selectedEntries, newEntry]);
    setAvailableKeys(availableKeys.filter((key) => key !== selectedKey));
  };

  const removeEntry = (entryId: string, keyToRemove: string) => {
    setSelectedEntries(selectedEntries.filter((entry) => entry.id !== entryId));
    setAvailableKeys([...availableKeys, keyToRemove].sort());
  };

  return (
    <div className="space-y-3">
      {/* Existing entries */}
      {selectedEntries.map((entry) => (
        <div key={entry.id} className="flex items-center space-x-3 rounded-lg border border-border p-3">
          <GuardrailField
            control={control}
            name={`${fullFieldKey}.${entry.key}`}
            label={entry.key}
            defaultValue={readRecord(value, entry.key)}
            className="flex-1"
          >
            {(fieldControl) => {
              if (field.dict_value_type === "number") {
                return (
                  <NumericalInput
                    id={fieldControl.id}
                    name={fieldControl.name}
                    step={1}
                    placeholder={`Enter ${entry.key} value`}
                    value={asText(fieldControl.value)}
                    onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                      fieldControl.onChange(toNumberValue(event.target.value))
                    }
                    onBlur={fieldControl.onBlur}
                    aria-invalid={fieldControl["aria-invalid"]}
                    aria-describedby={fieldControl["aria-describedby"]}
                  />
                );
              }

              if (field.dict_value_type === "boolean") {
                return <BooleanSelect control={fieldControl} placeholder={`Select ${entry.key} value`} />;
              }

              return (
                <Input
                  id={fieldControl.id}
                  name={fieldControl.name}
                  ref={fieldControl.ref}
                  placeholder={`Enter ${entry.key} value`}
                  value={asText(fieldControl.value)}
                  onChange={fieldControl.onChange}
                  onBlur={fieldControl.onBlur}
                  aria-invalid={fieldControl["aria-invalid"]}
                  aria-describedby={fieldControl["aria-describedby"]}
                />
              );
            }}
          </GuardrailField>
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:text-destructive/80"
            onClick={() => removeEntry(entry.id, entry.key)}
          >
            Remove
          </Button>
        </div>
      ))}

      {/* Add new entry */}
      {availableKeys.length > 0 && (
        <div className="mt-2 flex items-center space-x-3">
          <Select
            items={availableKeys.map((key) => ({ label: key, value: key }))}
            value={null}
            onValueChange={(next: string | null) => next && addEntry(next)}
          >
            <SelectTrigger className="w-50">
              <SelectValue placeholder="Select category to configure" />
            </SelectTrigger>
            <SelectContent>
              {availableKeys.map((key) => (
                <SelectItem key={key} value={key}>
                  {key}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="text-sm text-muted-foreground">Select a category to add threshold configuration</span>
        </div>
      )}
    </div>
  );
};

interface OptionalParamInputProps {
  descriptor: ProviderParam;
  fieldKey: string;
  control: GuardrailFieldControlProps;
}

const OptionalParamInput: React.FC<OptionalParamInputProps> = ({ descriptor, fieldKey, control }) => {
  const { id, value, onChange, onBlur, ref, name, ...aria } = control;

  if (descriptor.type === "select" && descriptor.options) {
    return (
      <Select
        items={descriptor.options.map((option) => ({ label: option, value: option }))}
        value={asText(value) || null}
        onValueChange={(next: string | null) => onChange(next)}
      >
        <SelectTrigger id={id} className="w-full" {...aria}>
          <SelectValue placeholder={descriptor.description} />
        </SelectTrigger>
        <SelectContent>
          {descriptor.options.map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }

  if (descriptor.type === "multiselect" && descriptor.options) {
    return (
      <MultiSelect
        id={id}
        options={descriptor.options.map((option) => ({ label: option, value: option }))}
        value={asStringArray(value)}
        onValueChange={onChange}
        placeholder={descriptor.description}
      />
    );
  }

  if (descriptor.type === "bool" || descriptor.type === "boolean") {
    return <BooleanSelect control={control} placeholder={descriptor.description} />;
  }

  if (descriptor.type === "number") {
    return (
      <NumericalInput
        id={id}
        name={name}
        step={1}
        placeholder={descriptor.description}
        value={asText(value)}
        onChange={(event: React.ChangeEvent<HTMLInputElement>) => onChange(toNumberValue(event.target.value))}
        onBlur={onBlur}
        {...aria}
      />
    );
  }

  if (isSecretKey(fieldKey)) {
    return (
      <PasswordInput
        id={id}
        name={name}
        ref={ref}
        placeholder={descriptor.description}
        value={asText(value)}
        onChange={onChange}
        onBlur={onBlur}
        {...aria}
      />
    );
  }

  return (
    <Input
      id={id}
      name={name}
      ref={ref}
      placeholder={descriptor.description}
      value={asText(value)}
      onChange={onChange}
      onBlur={onBlur}
      {...aria}
    />
  );
};

const GuardrailOptionalParams: React.FC<GuardrailOptionalParamsProps> = ({
  optionalParams,
  parentFieldKey,
  control,
  values,
}) => {
  const renderField = (fieldKey: string, field: ProviderParam) => {
    const fullFieldKey = `${parentFieldKey}.${fieldKey}`;
    const value = values?.[fieldKey];
    // Handle dict fields separately since they manage their own fields
    if (field.type === "dict" && field.dict_key_options) {
      return (
        <div key={fullFieldKey} className="mb-8 rounded-lg border border-border bg-muted/40 p-6">
          <div className="mb-4 text-base font-medium text-foreground">{fieldKey}</div>
          <p className="mb-4 text-sm text-muted-foreground">{field.description}</p>
          <DictField field={field} fullFieldKey={fullFieldKey} control={control} value={value} />
        </div>
      );
    }

    return (
      <div key={fullFieldKey} className="mb-8 rounded-lg border border-border bg-card p-6 shadow-xs">
        <GuardrailField
          control={control}
          name={fullFieldKey}
          label={<span className="text-base">{fieldKey}</span>}
          description={field.description}
          rules={field.required ? requiredRule(`${fieldKey} is required`) : undefined}
          defaultValue={value !== undefined ? value : field.default_value}
        >
          {(fieldControl) => <OptionalParamInput descriptor={field} fieldKey={fieldKey} control={fieldControl} />}
        </GuardrailField>
      </div>
    );
  };

  if (!optionalParams.fields || Object.keys(optionalParams.fields).length === 0) {
    return null;
  }

  return (
    <div className="guardrail-optional-params">
      <div className="mb-8 border-b border-border pb-4">
        <h3 className="mb-2 text-lg font-semibold text-foreground">Optional Parameters</h3>
        <p className="text-sm text-muted-foreground">
          {optionalParams.description || "Configure additional settings for this guardrail provider"}
        </p>
      </div>

      <div className="space-y-8">
        {Object.entries(optionalParams.fields).map(([fieldKey, field]) => renderField(fieldKey, field))}
      </div>
    </div>
  );
};

export default GuardrailOptionalParams;
