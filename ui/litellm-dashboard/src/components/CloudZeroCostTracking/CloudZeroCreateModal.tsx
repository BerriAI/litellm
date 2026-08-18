import { Modal } from "antd";
import { useEffect } from "react";
import { z } from "zod/v4";

import { useCloudZeroCreate } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import MessageManager from "@/components/molecules/message_manager";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Input } from "@/components/ui/input";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";

import { CloudZeroApiKeyInput, labelWithHint } from "./CloudZeroFormControls";
import { buildCloudZeroPayload, EMPTY_CLOUDZERO_FORM_VALUES, type CloudZeroFormValues } from "./cloudZeroPayload";

interface CloudZeroCreationModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
}

const createSchema = z.object({
  api_key: z.string().min(1, "Please enter your CloudZero API key"),
  connection_id: z.string().min(1, "Please enter your CloudZero connection ID"),
  timezone: z.string(),
});

export default function CloudZeroCreationModal({ open, onOk, onCancel }: CloudZeroCreationModalProps) {
  const { accessToken } = useAuthorized();
  const form = useZodForm(createSchema, { defaultValues: EMPTY_CLOUDZERO_FORM_VALUES });
  const createMutation = useCloudZeroCreate(accessToken || "");

  useEffect(() => {
    if (open) {
      form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
    }
  }, [open, form]);

  const handleSubmit = (values: CloudZeroFormValues) => {
    createMutation.mutate(buildCloudZeroPayload(values), {
      onSuccess: () => {
        MessageManager.success("CloudZero integration created successfully");
        form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
        onOk();
      },
      onError: (error: Error) => {
        MessageManager.error(error.message || "Failed to create CloudZero integration");
      },
    });
  };

  const handleCancel = () => {
    form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
    onCancel();
  };

  return (
    <Modal
      title="Create CloudZero Integration"
      open={open}
      onOk={() => void form.handleSubmit(handleSubmit)()}
      onCancel={handleCancel}
      confirmLoading={createMutation.isPending}
      okText={createMutation.isPending ? "Creating..." : "Create"}
      cancelText="Cancel"
      okButtonProps={{
        disabled: createMutation.isPending,
      }}
      cancelButtonProps={{
        disabled: createMutation.isPending,
      }}
    >
      <TooltipProvider>
        <form onSubmit={(event) => event.preventDefault()} noValidate>
          <FieldGroup>
            <FormField control={form.control} name="api_key" label="CloudZero API Key">
              {({ ref, ...field }) => (
                <CloudZeroApiKeyInput {...field} ref={ref} placeholder="Enter your CloudZero API key" />
              )}
            </FormField>
            <FormField control={form.control} name="connection_id" label="Connection ID">
              {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="Enter your CloudZero connection ID" />}
            </FormField>
            <FormField
              control={form.control}
              name="timezone"
              label={labelWithHint("Timezone", "Timezone for date handling (defaults to UTC if not provided)")}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="UTC" />}
            </FormField>
          </FieldGroup>
        </form>
      </TooltipProvider>
    </Modal>
  );
}
