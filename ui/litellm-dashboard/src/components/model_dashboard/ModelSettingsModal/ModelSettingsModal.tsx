"use client";

import { ConfigType, useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { StoreModelInDBParams, useStoreModelInDB } from "@/app/(dashboard)/hooks/storeModelInDB/useStoreModelInDB";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Button, Modal, Skeleton, Space, Typography } from "antd";
import { CircleHelp } from "lucide-react";
import React, { useEffect, useMemo } from "react";
import { useForm } from "react-hook-form";

interface ModelSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const ModelSettingsModal: React.FC<ModelSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { mutateAsync, isPending } = useStoreModelInDB();
  const { data: proxyConfigData, isLoading: isLoadingConfig, refetch } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  // Refetch config when modal opens to ensure we have the latest values
  useEffect(() => {
    if (isVisible) {
      refetch();
    }
  }, [isVisible, refetch]);

  // Compute initial values from fetched config data
  const initialValues = useMemo<StoreModelInDBParams>(() => {
    if (!proxyConfigData) {
      return {
        store_model_in_db: false,
      };
    }

    const storeModelField = proxyConfigData.find((field) => field.field_name === "store_model_in_db");

    return {
      store_model_in_db: storeModelField?.field_value ?? false,
    };
  }, [proxyConfigData]);

  const form = useForm<StoreModelInDBParams>({ defaultValues: initialValues, values: initialValues });

  const handleFormSubmit = async (formValues: StoreModelInDBParams) => {
    try {
      await mutateAsync(formValues, {
        onSuccess: () => {
          toast.success("Model storage settings updated successfully");
          refetch();
          onSuccess?.();
        },
        onError: (error) => {
          toast.fromError("Failed to save model storage settings: " + parseErrorMessage(error));
        },
      });
    } catch (error) {
      toast.fromError("Failed to save model storage settings: " + parseErrorMessage(error));
    }
  };

  const handleCancel = () => {
    form.reset(initialValues);
    onCancel();
  };

  return (
    <Modal
      title={<Typography.Title level={5}>Model Settings</Typography.Title>}
      open={isVisible}
      footer={
        <Space>
          <Button onClick={handleCancel} disabled={isPending || isLoadingConfig}>
            Cancel
          </Button>
          <Button
            type="primary"
            loading={isPending}
            disabled={isLoadingConfig}
            onClick={() => void form.handleSubmit(handleFormSubmit)()}
          >
            {isPending ? "Saving..." : "Save Settings"}
          </Button>
        </Space>
      }
      onCancel={handleCancel}
    >
      <TooltipProvider>
        <form onSubmit={(event) => event.preventDefault()}>
          <FieldGroup>
            <FormField
              control={form.control}
              name="store_model_in_db"
              label={labelWithHint(
                "Store Model in DB",
                proxyConfigData?.find((f) => f.field_name === "store_model_in_db")?.field_description ||
                  "If enabled, models and config are stored in and loaded from the database.",
              )}
            >
              {({ id, value, onChange, onBlur }) =>
                isLoadingConfig ? (
                  <Skeleton.Input active block />
                ) : (
                  <Switch
                    id={id}
                    checked={Boolean(value)}
                    onCheckedChange={onChange}
                    onBlur={onBlur}
                    className="w-fit"
                  />
                )
              }
            </FormField>
          </FieldGroup>
        </form>
      </TooltipProvider>
    </Modal>
  );
};

export default ModelSettingsModal;
