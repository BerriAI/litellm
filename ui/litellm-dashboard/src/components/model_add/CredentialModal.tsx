import { Input } from "@/components/ui/input";
import { Select as AntdSelect, Button, Modal, Tooltip, Typography } from "antd";
import type { UploadProps } from "antd/es/upload";
import { useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { antdRequired } from "../common_components/antdFormRules";
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

const { Link } = Typography;

interface CredentialModalProps {
  open: boolean;
  onCancel: () => void;
  onSubmit: (values: any) => void;
  uploadProps: UploadProps;
  mode: "add" | "edit";
  existingCredential?: CredentialItem | null;
}

export default function CredentialModal({
  open,
  onCancel,
  onSubmit,
  uploadProps,
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
    const values = projectMountedValues(registry, form.getValues());
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
    <Modal
      title={isEdit ? "Edit Credential" : "Add New Credential"}
      open={open}
      onCancel={closeAndReset}
      footer={null}
      width={600}
      destroyOnHidden={isEdit}
    >
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
              rules={{ validate: { required: antdRequired("Credential name is required") } }}
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
              rules={{ validate: { required: antdRequired("Required") } }}
              className="mb-4"
            >
              {(control) => (
                <AntdSelect
                  id={control.id}
                  showSearch
                  value={control.value as string | undefined}
                  onBlur={control.onBlur}
                  onChange={(value) => {
                    control.onChange(value);
                    resetCredentialFormOnProviderChange(formAdapter, value as Providers, setSelectedProvider);
                  }}
                >
                  {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => (
                    <AntdSelect.Option key={providerEnum} value={providerEnum}>
                      <div className="flex items-center space-x-2">
                        <Logo provider={providerEnum} label={providerDisplayName} className="w-5 h-5" />
                        <span>{providerDisplayName}</span>
                      </div>
                    </AntdSelect.Option>
                  ))}
                </AntdSelect>
              )}
            </MountedFormField>

            <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />

            <div className="flex justify-between items-center">
              <Tooltip title="Get help on our github">
                <Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Link>
              </Tooltip>

              <div>
                <Button onClick={closeAndReset} style={{ marginRight: 10 }}>
                  Cancel
                </Button>
                <Button htmlType="submit">{isEdit ? "Update Credential" : "Add Credential"}</Button>
              </div>
            </div>
          </form>
        </MountedFormProvider>
      </FormProvider>
    </Modal>
  );
}
