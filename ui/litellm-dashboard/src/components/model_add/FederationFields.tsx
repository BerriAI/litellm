import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { isMaskedSecret } from "@/utils/maskedSecretUtils";
import { MountedFormField, type MountedFieldControlProps } from "../common_components/MountedFormField";
import {
  FEDERATION_CORE_FIELDS,
  identitySourceById,
  identitySourceOptions,
  validateFederationValueStored,
  validateIdentityTokenReference,
  validateIssuerTtlSeconds,
  type IdentitySourceId,
} from "./anthropic_federation";
import type { FederatedSelection } from "./credential_federation";
import { requiredFederationValue, validateMaskedValueUntouched, type FederationField } from "./federation_field";
import { OPENAI_FEDERATION_FIELDS } from "./openai_federation";

type FieldValidator = (value: unknown, formValues: Record<string, unknown>) => string | true;

interface FederationFieldsProps {
  selection: FederatedSelection;
  onIdentitySourceChange: (identitySource: IdentitySourceId) => void;
  storedValues: Record<string, unknown>;
}

const IDENTITY_SOURCE_SELECT_ID = "anthropic_federation_identity_source";
const STORED_VALUE_MESSAGE_FIELD_KEY = FEDERATION_CORE_FIELDS[0].key;

const anthropicValidators = (
  field: FederationField,
  identitySource: IdentitySourceId,
): Readonly<Record<string, FieldValidator>> => ({
  ...(field.key === STORED_VALUE_MESSAGE_FIELD_KEY ? { stored: validateFederationValueStored(identitySource) } : {}),
  ...(field.key === "anthropic_identity_token" ? { reference: validateIdentityTokenReference } : {}),
  ...(field.control === "integer" ? { ttl: validateIssuerTtlSeconds } : {}),
});

const selectItems = (field: FederationField, value: unknown) => [
  ...(isMaskedSecret(value) ? [{ value: value as string, label: `Stored: ${value as string}` }] : []),
  ...(field.options ?? []).map((option) => ({ value: option, label: option })),
];

const renderControl = (field: FederationField, control: MountedFieldControlProps) => {
  if (field.control === "select") {
    const items = selectItems(field, control.value);
    return (
      <Select
        items={items}
        value={typeof control.value === "string" && control.value !== "" ? control.value : null}
        onValueChange={(value) => control.onChange(value ?? "")}
      >
        <SelectTrigger id={control.id} onBlur={control.onBlur} className="w-full">
          <SelectValue placeholder="Proxy default" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={null}>Proxy default</SelectItem>
          {items.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }
  return (
    <Input
      id={control.id}
      value={control.value === undefined || control.value === null ? "" : String(control.value)}
      onChange={control.onChange}
      onBlur={control.onBlur}
      placeholder={field.placeholder}
      inputMode={field.control === "integer" ? "numeric" : undefined}
      aria-invalid={control["aria-invalid"]}
    />
  );
};

interface FederationFieldInputProps {
  field: FederationField;
  storedValue: unknown;
  validators?: Readonly<Record<string, FieldValidator>>;
}

function FederationFieldInput({ field, storedValue, validators }: FederationFieldInputProps) {
  return (
    <MountedFormField
      label={labelWithHint(field.label, field.tooltip)}
      name={field.key}
      required={field.required}
      rules={{
        validate: {
          ...(field.required ? { required: requiredFederationValue } : {}),
          ...validators,
          masked: validateMaskedValueUntouched(storedValue),
        },
      }}
      className="mb-4"
    >
      {(control) => renderControl(field, control)}
    </MountedFormField>
  );
}

interface AnthropicFederationFieldsProps {
  identitySource: IdentitySourceId;
  onIdentitySourceChange: (identitySource: IdentitySourceId) => void;
  storedValues: Record<string, unknown>;
}

function AnthropicFederationFields({
  identitySource,
  onIdentitySourceChange,
  storedValues,
}: AnthropicFederationFieldsProps) {
  const identitySourceItems = identitySourceOptions(storedValues);
  const renderField = (field: FederationField) => (
    <FederationFieldInput
      key={field.key}
      field={field}
      storedValue={storedValues[field.key]}
      validators={anthropicValidators(field, identitySource)}
    />
  );

  return (
    <>
      {FEDERATION_CORE_FIELDS.map(renderField)}
      <div className="mb-4 flex flex-col gap-2">
        <label htmlFor={IDENTITY_SOURCE_SELECT_ID} className="text-sm font-medium">
          {labelWithHint(
            "Identity Source",
            "Where the proxy gets the identity token it exchanges with Anthropic for an access token.",
          )}
        </label>
        <Select
          items={identitySourceItems}
          value={identitySource}
          onValueChange={(value) => onIdentitySourceChange(value as IdentitySourceId)}
        >
          <SelectTrigger id={IDENTITY_SOURCE_SELECT_ID} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {identitySourceItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {identitySource === "unrecognized" && (
          <p className="text-sm text-muted-foreground">
            This form does not offer the stored identity source. Saving keeps it and its settings as stored. Pick
            another source to replace it.
          </p>
        )}
        {identitySource === "environment" && (
          <p className="text-sm text-muted-foreground">
            The proxy reads ANTHROPIC_IDENTITY_TOKEN_FILE or ANTHROPIC_IDENTITY_TOKEN from its own environment.
          </p>
        )}
      </div>
      {identitySourceById(identitySource).fields.map(renderField)}
    </>
  );
}

function OpenAIFederationFields({ storedValues }: { storedValues: Record<string, unknown> }) {
  return (
    <>
      <p className="mb-4 text-sm text-muted-foreground">
        The proxy exchanges the identity token for an OpenAI access token only when OPENAI_API_KEY is unset in its
        environment. Otherwise it sends that key instead.
      </p>
      {OPENAI_FEDERATION_FIELDS.map((field) => (
        <FederationFieldInput key={field.key} field={field} storedValue={storedValues[field.key]} />
      ))}
    </>
  );
}

export default function FederationFields({ selection, onIdentitySourceChange, storedValues }: FederationFieldsProps) {
  switch (selection.provider) {
    case "anthropic":
      return (
        <AnthropicFederationFields
          identitySource={selection.identitySource}
          onIdentitySourceChange={onIdentitySourceChange}
          storedValues={storedValues}
        />
      );
    case "openai":
      return <OpenAIFederationFields storedValues={storedValues} />;
  }
}
