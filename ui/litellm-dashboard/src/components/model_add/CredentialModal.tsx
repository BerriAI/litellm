import type { ParseKeys } from "i18next";
import { Input } from "@/components/ui/input";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { useTranslation } from "react-i18next";
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
import {
  displayNameChange,
  initialFormValues,
  resetCredentialFormOnProviderChange,
  withoutRestrictedFields,
  type CredentialSubmission,
} from "./credential_form_helpers";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import FederationFields from "./FederationFields";
import InternalIssuerJwks from "./InternalIssuerJwks";
import { DEFAULT_IDENTITY_SOURCE, inferIdentitySource, type IdentitySourceId } from "./anthropic_federation";
import {
  buildCreateCredentialValues,
  buildCredentialPatch,
  buildProviderChangePatch,
  federatedProviderOf,
  inferAuthMethod,
  isFederatedCredential,
  jwksPanelFor,
  providerFieldValidators,
  selectionFor,
  type AuthMethod,
} from "./credential_federation";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const providerOptions: SearchSelectOption[] = Object.entries(Providers).map(([providerEnum, providerDisplayName]) => ({
  label: providerDisplayName,
  value: providerEnum,
  icon: <Logo provider={providerEnum} label={providerDisplayName} className="w-5 h-5" />,
}));

const AUTH_METHOD_SELECT_ID = "credential_auth_method";
const API_KEY_FIELDS: readonly string[] = ["api_key"];
const NO_HIDDEN_FIELDS: readonly string[] = [];

const authMethodItems: { value: AuthMethod; labelKey: ParseKeys }[] = [
  { value: "api_key", labelKey: "modelAdd.credentialModal.authMethods.apiKey" },
  { value: "federation", labelKey: "modelAdd.credentialModal.authMethods.federation" },
];

interface CredentialModalProps {
  open: boolean;
  onCancel: () => void;
  onSubmit: (submission: CredentialSubmission, valuesToDelete: readonly string[]) => void;
  mode: "add" | "edit";
  existingCredential?: CredentialItem | null;
  initialProvider?: string | null;
  initialAuthMethod?: AuthMethod;
  initialAuthTypeId?: string;
  providerLocked?: boolean;
}

const sameProvider = (left: string | null | undefined, right: string | null | undefined): boolean =>
  (left ?? "").toLowerCase() === (right ?? "").toLowerCase();

export default function CredentialModal({
  open,
  onCancel,
  onSubmit,
  mode,
  existingCredential = null,
  initialProvider = null,
  initialAuthMethod,
  initialAuthTypeId,
  providerLocked = false,
}: CredentialModalProps) {
  const { t } = useTranslation();
  const isEdit = mode === "edit";
  const [selectedProvider, setSelectedProvider] = useState<string | null>(
    (existingCredential?.credential_info.custom_llm_provider as Providers) ?? initialProvider ?? Providers.OpenAI,
  );
  const storedProvider = existingCredential?.credential_info.custom_llm_provider ?? null;
  const storedValues: Record<string, unknown> = existingCredential?.credential_values ?? {};
  const storedAuthMethod = inferAuthMethod(storedValues);
  const storedIdentitySource = isFederatedCredential(storedValues)
    ? inferIdentitySource(storedValues)
    : DEFAULT_IDENTITY_SOURCE;
  const storedSelection = selectionFor(federatedProviderOf(storedProvider), storedAuthMethod, storedIdentitySource);
  const [authMethod, setAuthMethod] = useState<AuthMethod>(
    existingCredential ? storedAuthMethod : initialAuthMethod ?? "api_key",
  );
  const [identitySource, setIdentitySource] = useState<IdentitySourceId>(storedIdentitySource);
  const selection = selectionFor(federatedProviderOf(selectedProvider), authMethod, identitySource);

  const initialValues = initialFormValues(existingCredential, initialProvider);

  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues: initialValues });
  const registry = useMountRegistry();

  const formAdapterFor = (provider: string | null) => ({
    getFieldValue: (field: string) => form.getValues(field),
    resetFields: () => form.reset(isEdit && !sameProvider(provider, storedProvider) ? {} : initialValues),
    setFieldValue: (field: string, value: unknown) => form.setValue(field, value),
  });

  const changeProvider = (provider: string | null) => {
    const backToStored = isEdit && sameProvider(provider, storedProvider);
    setAuthMethod(backToStored ? storedAuthMethod : "api_key");
    setIdentitySource(backToStored ? storedIdentitySource : DEFAULT_IDENTITY_SOURCE);
    resetCredentialFormOnProviderChange(formAdapterFor(provider), provider, setSelectedProvider);
  };

  const changeAuthMethod = (method: AuthMethod) => {
    form.clearErrors(Object.keys(providerFieldValidators(selection)));
    setAuthMethod(method);
  };

  const handleSubmit = async () => {
    const isValid = await form.trigger(registry.mountedNames() as string[]);
    if (!isValid) {
      return;
    }
    const values = projectMountedValues(registry, form.getValues);
    const meta = {
      credential_name: String(values.credential_name),
      custom_llm_provider: String(values.custom_llm_provider),
      ...displayNameChange(values.display_name, existingCredential),
    };
    if (!isEdit) {
      onSubmit(
        { ...meta, credential_values: buildCreateCredentialValues(withoutRestrictedFields(values), selection) },
        [],
      );
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
    onSubmit({ ...meta, credential_values: patch.credential_values }, valuesToDelete);
  };

  const closeAndReset = () => {
    onCancel();
    form.reset();
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && closeAndReset()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>
            {isEdit ? t("modelAdd.credentialModal.editTitle") : t("modelAdd.credentialModal.addTitle")}
          </DialogTitle>
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
                label={t("modelAdd.credentialModal.credentialName")}
                name="credential_name"
                required
                rules={{ validate: { required: requiredRule(t("modelAdd.credentialModal.credentialNameRequired")) } }}
                className="mb-4"
              >
                {(control) => (
                  <Input
                    id={control.id}
                    value={typeof control.value === "string" ? control.value : ""}
                    onChange={control.onChange}
                    onBlur={control.onBlur}
                    placeholder={t("modelAdd.credentialModal.credentialNamePlaceholder")}
                    disabled={isEdit}
                  />
                )}
              </MountedFormField>

              <MountedFormField label={t("modelAdd.credentialModal.displayName")} name="display_name" className="mb-4">
                {(control) => (
                  <Input
                    id={control.id}
                    value={typeof control.value === "string" ? control.value : ""}
                    onChange={control.onChange}
                    onBlur={control.onBlur}
                    placeholder={t("modelAdd.credentialModal.displayNamePlaceholder")}
                  />
                )}
              </MountedFormField>

              <MountedFormField
                label={labelWithHint(
                  t("modelAdd.credentialModal.provider"),
                  t("modelAdd.credentialModal.providerHint"),
                )}
                name="custom_llm_provider"
                required
                rules={{ validate: { required: requiredRule(t("modelAdd.credentialModal.required")) } }}
                className="mb-4"
              >
                {(control) => (
                  <SearchSelect
                    inputId={control.id}
                    placeholder={t("modelAdd.credentialModal.selectProvider")}
                    options={providerOptions}
                    value={typeof control.value === "string" ? control.value : null}
                    disabled={providerLocked}
                    onValueChange={(value) => {
                      control.onChange(value);
                      changeProvider(value);
                    }}
                  />
                )}
              </MountedFormField>

              {federatedProviderOf(selectedProvider) !== null && (
                <div className="mb-4 flex flex-col gap-2">
                  <label htmlFor={AUTH_METHOD_SELECT_ID} className="text-sm font-medium">
                    {labelWithHint(
                      t("modelAdd.credentialModal.authentication"),
                      t("modelAdd.credentialModal.authenticationHint"),
                    )}
                  </label>
                  <Select
                    items={authMethodItems.map((item) => ({ value: item.value, label: t(item.labelKey) }))}
                    value={authMethod}
                    onValueChange={(value) => changeAuthMethod(value as AuthMethod)}
                  >
                    <SelectTrigger id={AUTH_METHOD_SELECT_ID} className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {authMethodItems.map((item) => (
                        <SelectItem key={item.value} value={item.value}>
                          {t(item.labelKey)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              <ProviderSpecificFields
                selectedProvider={selectedProvider}
                hiddenFieldKeys={selection.authMethod === "federation" ? API_KEY_FIELDS : NO_HIDDEN_FIELDS}
                initialAuthTypeId={initialAuthTypeId}
                fieldValidators={providerFieldValidators(selection)}
              />

              {selection.authMethod === "federation" && (
                <FederationFields
                  selection={selection}
                  onIdentitySourceChange={setIdentitySource}
                  storedValues={storedValues}
                />
              )}

              <InternalIssuerJwks panel={jwksPanelFor(existingCredential, selection)} />

              <div className="flex justify-between items-center">
                <SimpleTooltip content={t("modelAdd.credentialModal.helpTooltip")}>
                  <a href="https://github.com/BerriAI/litellm/issues" className="text-sm text-primary hover:underline">
                    {t("modelAdd.credentialModal.needHelp")}
                  </a>
                </SimpleTooltip>

                <div>
                  <Button variant="outline" className="mr-2.5" onClick={closeAndReset}>
                    {t("modelAdd.credentialModal.cancel")}
                  </Button>
                  <Button type="submit">
                    {isEdit ? t("modelAdd.credentialModal.updateSubmit") : t("modelAdd.credentialModal.addSubmit")}
                  </Button>
                </div>
              </div>
            </form>
          </MountedFormProvider>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}
