import { Input } from "@/components/ui/input";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
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
import { resetCredentialFormOnProviderChange } from "./credential_form_helpers";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const providerOptions: SearchSelectOption[] = Object.entries(Providers).map(([providerEnum, providerDisplayName]) => ({
  label: providerDisplayName,
  value: providerEnum,
  icon: <Logo provider={providerEnum} label={providerDisplayName} className="w-5 h-5" />,
}));

interface CredentialModalProps {
  open: boolean;
  onCancel: () => void;
  onSubmit: (values: any) => void;
  mode: "add" | "edit";
  existingCredential?: CredentialItem | null;
}

export default function CredentialModal({
  open,
  onCancel,
  onSubmit,
  mode,
  existingCredential = null,
}: CredentialModalProps) {
  const isEdit = mode === "edit";
  const [selectedProvider, setSelectedProvider] = useState<Providers>(
    (existingCredential?.credential_info.custom_llm_provider as Providers) ?? Providers.OpenAI,
  );

  const initialValues = existingCredential
    ? {
        credential_name: existingCredential.credential_name,
        custom_llm_provider: existingCredential.credential_info.custom_llm_provider,
        ...Object.fromEntries(
          Object.entries(existingCredential.credential_values || {}).map(([key, value]) => [key, value ?? null]),
        ),
      }
    : undefined;

  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues: initialValues });
  const registry = useMountRegistry();

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
    onSubmit(filteredValues);
    form.reset();
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

              <ProviderSpecificFields selectedProvider={selectedProvider} />

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
