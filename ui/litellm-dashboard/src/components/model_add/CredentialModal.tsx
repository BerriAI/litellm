import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldDescription, FieldLegend, FieldSet } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Logo } from "@/components/molecules/logo/Logo";
import CopyButton from "@/components/shared/CopyButton";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useQuery } from "@tanstack/react-query";
import { CircleCheck, CircleX } from "lucide-react";
import { useState } from "react";
import { FormProvider, useForm, useWatch } from "react-hook-form";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { requiredRule } from "../common_components/formRules";
import {
  MountedFormField,
  MountedFormProvider,
  projectMountedValues,
  useMountRegistry,
  type MountedFormValues,
} from "../common_components/MountedFormField";
import type {
  AnthropicJwks,
  CredentialItem,
  ProviderModelDiscoveryRequest,
  ProviderModelDiscoveryResponse,
} from "../networking";
import { Providers } from "../provider_info_helpers";
import {
  computeCredentialValuesToDelete,
  planCredentialTest,
  providerEnumKey,
  resetCredentialFormOnProviderChange,
  summarizeDiscoveredModels,
} from "./credential_form_helpers";

const providerOptions: SearchSelectOption[] = Object.entries(Providers).map(([providerEnum, providerDisplayName]) => ({
  label: providerDisplayName,
  value: providerEnum,
  icon: <Logo provider={providerEnum} label={providerDisplayName} className="w-5 h-5" />,
}));

type ConnectionTest =
  | { readonly kind: "idle" }
  | { readonly kind: "testing" }
  | { readonly kind: "success"; readonly message: string }
  | { readonly kind: "failure"; readonly message: string };

interface CredentialModalProps {
  open: boolean;
  onCancel: () => void;
  onSubmit: (values: any, credentialValuesToDelete: string[]) => Promise<boolean>;
  mode: "add" | "edit";
  existingCredential?: CredentialItem | null;
  initialProvider?: Providers;
  initialVariantId?: string;
  testConnection: (request: ProviderModelDiscoveryRequest) => Promise<ProviderModelDiscoveryResponse>;
  loadJwks: (credentialName: string) => Promise<AnthropicJwks>;
}

function JwksExport({
  credentialName,
  loadJwks,
}: {
  credentialName: string;
  loadJwks: CredentialModalProps["loadJwks"];
}) {
  const jwks = useQuery({ queryKey: ["credential-jwks", credentialName], queryFn: () => loadJwks(credentialName) });
  const jwksText = jwks.data ? JSON.stringify(jwks.data, null, 2) : null;
  return (
    <FieldSet className="mb-4">
      <FieldLegend variant="label">Public JWKS</FieldLegend>
      <FieldDescription>
        In the Claude Console, open Settings, then Workload identity, click Connect workload, choose Custom OIDC and
        paste this key set as the inline JWKS together with the Issuer URL and Subject above. Copy the Organization ID
        and Federation Rule ID the Console shows into the fields above, update the credential, then test the connection.
      </FieldDescription>
      {jwks.isPending && <p className="text-sm text-muted-foreground">Loading JWKS...</p>}
      {jwks.isError && <p className="text-sm text-destructive">{extractProxyErrorMessage(jwks.error)}</p>}
      {jwksText && (
        <div className="relative rounded-md border bg-muted p-3">
          <CopyButton value={jwksText} label="Copy JWKS" className="absolute top-2 right-2" />
          <pre className="overflow-x-auto text-xs">{jwksText}</pre>
        </div>
      )}
    </FieldSet>
  );
}

export default function CredentialModal({
  open,
  onCancel,
  onSubmit,
  mode,
  existingCredential = null,
  initialProvider,
  initialVariantId,
  testConnection,
  loadJwks,
}: CredentialModalProps) {
  const isEdit = mode === "edit";
  const [selectedProvider, setSelectedProvider] = useState<Providers>(
    (existingCredential?.credential_info.custom_llm_provider as Providers) ?? initialProvider ?? Providers.OpenAI,
  );
  const [connectionTest, setConnectionTest] = useState<ConnectionTest>({ kind: "idle" });

  const presetValues = initialProvider ? { custom_llm_provider: providerEnumKey(initialProvider) } : undefined;
  const initialValues = existingCredential
    ? {
        credential_name: existingCredential.credential_name,
        custom_llm_provider: existingCredential.credential_info.custom_llm_provider,
        ...Object.fromEntries(
          Object.entries(existingCredential.credential_values || {}).map(([key, value]) => [key, value ?? null]),
        ),
      }
    : presetValues;

  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues: initialValues });
  const registry = useMountRegistry();
  useWatch({ control: form.control });
  const testInput = {
    mode,
    provider: selectedProvider,
    credentialName: existingCredential?.credential_name ?? "",
    mountedValues: projectMountedValues(registry, form.getValues),
    hasUnsavedChanges: form.formState.isDirty,
  };
  const testPlan = planCredentialTest(testInput);
  const jwksCredentialName =
    isEdit && existingCredential?.credential_values?.anthropic_identity_source === "internal_issuer"
      ? existingCredential.credential_name
      : null;

  const formAdapter = {
    getFieldValue: (field: string) => form.getValues(field),
    resetFields: () => form.reset(),
    setFieldValue: (field: string, value: unknown) => form.setValue(field, value),
  };

  const handleSubmit = async () => {
    const isValid = await form.trigger(registry.mountedNames() as string[]);
    if (!isValid) {
      return;
    }
    const values = projectMountedValues(registry, form.getValues);
    const filteredValues = Object.entries(values).reduce((acc, [key, value]) => {
      if (value !== "" && value !== undefined && value !== null) {
        acc[key] = value;
      }
      return acc;
    }, {} as any);
    const credentialValuesToDelete = isEdit
      ? computeCredentialValuesToDelete(existingCredential?.credential_values ?? {}, values)
      : [];
    const saved = await onSubmit(filteredValues, credentialValuesToDelete);
    if (saved) {
      form.reset();
    }
  };

  const handleTestConnection = async () => {
    if (testPlan.kind !== "ready") {
      return;
    }
    setConnectionTest({ kind: "testing" });
    try {
      const { models } = await testConnection(testPlan.request);
      setConnectionTest({ kind: "success", message: summarizeDiscoveredModels(models) });
    } catch (error) {
      setConnectionTest({ kind: "failure", message: extractProxyErrorMessage(error) });
    }
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
                    value={(control.value as string | undefined) ?? ""}
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
                    value={(control.value as string | undefined) ?? ""}
                    onValueChange={(value) => {
                      control.onChange(value);
                      resetCredentialFormOnProviderChange(formAdapter, value as Providers, setSelectedProvider);
                    }}
                  />
                )}
              </MountedFormField>

              <ProviderSpecificFields selectedProvider={selectedProvider} initialVariantId={initialVariantId} />

              {jwksCredentialName && <JwksExport credentialName={jwksCredentialName} loadJwks={loadJwks} />}

              {connectionTest.kind === "success" && (
                <Alert className="mb-4">
                  <CircleCheck />
                  <AlertDescription>{connectionTest.message}</AlertDescription>
                </Alert>
              )}
              {connectionTest.kind === "failure" && (
                <Alert variant="destructive" className="mb-4">
                  <CircleX />
                  <AlertDescription>{connectionTest.message}</AlertDescription>
                </Alert>
              )}
              {testPlan.kind === "unavailable" && (
                <p className="mb-2 text-right text-xs text-muted-foreground">{testPlan.reason}</p>
              )}

              <div className="flex justify-between items-center">
                <div className="flex items-center gap-4">
                  <Button type="button" variant="outline" onClick={closeAndReset}>
                    Cancel
                  </Button>
                  <SimpleTooltip content="Get help on our github">
                    <a
                      href="https://github.com/BerriAI/litellm/issues"
                      className="text-sm text-primary hover:underline"
                    >
                      Need Help?
                    </a>
                  </SimpleTooltip>
                </div>

                <div className="flex items-center gap-2.5">
                  <Button
                    type="button"
                    variant="outline"
                    disabled={testPlan.kind !== "ready" || connectionTest.kind === "testing"}
                    onClick={() => void handleTestConnection()}
                  >
                    {connectionTest.kind === "testing" ? "Testing..." : "Test Connection"}
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
