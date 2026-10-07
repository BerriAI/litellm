import { Input } from "@/components/ui/input";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { authTypeFieldKeys } from "../add_model/provider_auth_types";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { requiredRule } from "../common_components/formRules";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import {
  MountedFormField,
  MountedFormProvider,
  projectMountedValues,
  useMountRegistry,
  type MountedFormValues,
} from "../common_components/MountedFormField";
import { CredentialItem } from "../networking";
import { Providers } from "../provider_info_helpers";
import { Logo } from "@/components/molecules/logo/Logo";
import { resetCredentialFormOnProviderChange, withoutRestrictedFields } from "./credential_form_helpers";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import AnthropicFederationFields from "./AnthropicFederationFields";
import {
  buildCreateCredentialValues,
  buildCredentialPatch,
  buildProviderChangePatch,
  inferAuthMethod,
  inferIdentitySource,
  isAnthropicProvider,
  isFederatedCredential,
  type AnthropicAuthMethod,
  type IdentitySourceId,
} from "./anthropic_federation";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const providerOptions: SearchSelectOption[] = Object.entries(Providers).map(([providerEnum, providerDisplayName]) => ({
  label: providerDisplayName,
  value: providerEnum,
  icon: <Logo provider={providerEnum} label={providerDisplayName} className="w-5 h-5" />,
}));

const AUTH_METHOD_SELECT_ID = "anthropic_auth_method";
const API_KEY_FIELDS: readonly string[] = ["api_key"];
const NO_HIDDEN_FIELDS: readonly string[] = [];

const authMethodItems: { value: AnthropicAuthMethod; label: string }[] = [
  { value: "api_key", label: "API key" },
  { value: "federation", label: "Workload identity federation" },
];

interface CredentialModalProps {
  open: boolean;
  onCancel: () => void;
  onSubmit: (values: Record<string, unknown>, valuesToDelete: readonly string[]) => void;
  mode: "add" | "edit";
  existingCredential?: CredentialItem | null;
  initialProvider?: string | null;
  initialAuthMethod?: AnthropicAuthMethod;
  providerLocked?: boolean;
}

const sameProvider = (left: string | null | undefined, right: string | null | undefined): boolean =>
  (left ?? "").toLowerCase() === (right ?? "").toLowerCase();

const initialFormValues = (
  existingCredential: CredentialItem | null | undefined,
  initialProvider: string | null | undefined,
): MountedFormValues | undefined => {
  if (existingCredential) {
    return {
      credential_name: existingCredential.credential_name,
      custom_llm_provider: existingCredential.credential_info.custom_llm_provider,
      ...Object.fromEntries(
        Object.entries(existingCredential.credential_values || {}).map(([key, value]) => [key, value ?? null]),
      ),
    };
  }
  return initialProvider ? { custom_llm_provider: initialProvider } : undefined;
};

export default function CredentialModal({
  open,
  onCancel,
  onSubmit,
  mode,
  existingCredential = null,
  initialProvider = null,
  initialAuthMethod,
  providerLocked = false,
}: CredentialModalProps) {
  const isEdit = mode === "edit";
  const [selectedProvider, setSelectedProvider] = useState<string | null>(
    (existingCredential?.credential_info.custom_llm_provider as Providers) ?? initialProvider ?? Providers.OpenAI,
  );
  const storedProvider = existingCredential?.credential_info.custom_llm_provider ?? null;
  const storedValues: Record<string, unknown> = existingCredential?.credential_values ?? {};
  const storedSelection = {
    authMethod: inferAuthMethod(storedValues),
    identitySource: inferIdentitySource(storedValues),
  };
  const [authMethod, setAuthMethod] = useState<AnthropicAuthMethod>(
    existingCredential ? storedSelection.authMethod : initialAuthMethod ?? "api_key",
  );
  const [identitySource, setIdentitySource] = useState<IdentitySourceId>(
    isFederatedCredential(storedValues) ? storedSelection.identitySource : "token_file",
  );
  const isAnthropic = isAnthropicProvider(selectedProvider);
  const selection = { authMethod: isAnthropic ? authMethod : ("api_key" as const), identitySource };

  const initialValues = initialFormValues(existingCredential, initialProvider);

  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues: initialValues });
  const registry = useMountRegistry();

  const formAdapterFor = (provider: string | null) => ({
    getFieldValue: (field: string) => form.getValues(field),
    resetFields: () => form.reset(isEdit && !sameProvider(provider, storedProvider) ? {} : initialValues),
    setFieldValue: (field: string, value: unknown) => form.setValue(field, value),
  });

  const handleSubmit = async () => {
    const isValid = await form.trigger(registry.mountedNames() as string[]);
    if (!isValid) {
      return;
    }
    const values = projectMountedValues(registry, form.getValues);
    const meta = {
      credential_name: values.credential_name,
      custom_llm_provider: values.custom_llm_provider,
    };
    if (!isEdit) {
      onSubmit({ ...meta, ...buildCreateCredentialValues(withoutRestrictedFields(values), selection) }, []);
      return;
    }
    const sameStoredProvider = sameProvider(selectedProvider, storedProvider);
    const projectedValues = withoutRestrictedFields(values);
    const patch = sameStoredProvider
      ? buildCredentialPatch(storedValues, projectedValues, storedSelection, selection)
      : buildProviderChangePatch(storedValues, projectedValues, selection);
    const authTypeValuesToDelete = sameStoredProvider
      ? authTypeFieldKeys(selectedProvider).filter((key) => key in storedValues && !(key in projectedValues))
      : [];
    const valuesToDelete = Array.from(new Set([...patch.credential_values_to_delete, ...authTypeValuesToDelete]));
    onSubmit({ ...meta, ...patch.credential_values }, valuesToDelete);
  };

  const closeAndReset = () => {
    onCancel();
    form.reset();
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && closeAndReset()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>{isEdit ? "Edit Credential" : "Add New Credential"}</DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <MountedFormProvider value={{ control: form.control, registry }}>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void handleSubmit();
              }}
            >
              <MountedFormField
                label="Credential Name:"
                name="credential_name"
                required
                rules={{ validate: { required: requiredRule("Credential name is required") } }}
                className="mb-4"
              >
                {(control) => (
                  <Input
                    id={control.id}
                    value={typeof control.value === "string" ? control.value : ""}
                    onChange={control.onChange}
                    onBlur={control.onBlur}
                    placeholder="Enter a friendly name for these credentials"
                    disabled={isEdit}
                  />
                )}
              </MountedFormField>

              <MountedFormField
                label={labelWithHint("Provider:", "Helper to auto-populate provider specific fields")}
                name="custom_llm_provider"
                required
                rules={{ validate: { required: requiredRule("Required") } }}
                className="mb-4"
              >
                {(control) => (
                  <SearchSelect
                    inputId={control.id}
                    placeholder="Select a provider"
                    options={providerOptions}
                    value={typeof control.value === "string" ? control.value : null}
                    disabled={providerLocked}
                    onValueChange={(value) => {
                      control.onChange(value);
                      resetCredentialFormOnProviderChange(formAdapterFor(value), value, setSelectedProvider);
                    }}
                  />
                )}
              </MountedFormField>

              {isAnthropic && (
                <div className="mb-4 flex flex-col gap-2">
                  <label htmlFor={AUTH_METHOD_SELECT_ID} className="text-sm font-medium">
                    {labelWithHint(
                      "Authentication:",
                      "Workload identity federation exchanges an identity token for a short-lived Anthropic access token, so no API key is stored.",
                    )}
                  </label>
                  <Select
                    items={authMethodItems}
                    value={authMethod}
                    onValueChange={(value) => setAuthMethod(value as AnthropicAuthMethod)}
                  >
                    <SelectTrigger id={AUTH_METHOD_SELECT_ID} className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {authMethodItems.map((item) => (
                        <SelectItem key={item.value} value={item.value}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              <ProviderSpecificFields
                selectedProvider={selectedProvider}
                hiddenFieldKeys={selection.authMethod === "federation" ? API_KEY_FIELDS : NO_HIDDEN_FIELDS}
              />

              {selection.authMethod === "federation" && (
                <AnthropicFederationFields
                  identitySource={identitySource}
                  onIdentitySourceChange={setIdentitySource}
                  storedValues={storedValues}
                />
              )}

              <div className="flex justify-between items-center">
                <SimpleTooltip content="Get help on our github">
                  <a href="https://github.com/BerriAI/litellm/issues" className="text-sm text-primary hover:underline">
                    Need Help?
                  </a>
                </SimpleTooltip>

                <div>
                  <Button variant="outline" className="mr-2.5" onClick={closeAndReset}>
                    Cancel
                  </Button>
                  <Button type="submit">{isEdit ? "Update Credential" : "Add Credential"}</Button>
                </div>
              </div>
            </form>
          </MountedFormProvider>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}
