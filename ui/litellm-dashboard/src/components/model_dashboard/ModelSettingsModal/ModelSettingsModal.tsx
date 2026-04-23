"use client";

import {
  ConfigType,
  useProxyConfig,
} from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import {
  StoreModelInDBParams,
  useStoreModelInDB,
} from "@/app/(dashboard)/hooks/storeModelInDB/useStoreModelInDB";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Form, Switch as AntdFormSwitch } from "antd";
import React, { useEffect, useMemo } from "react";

interface ModelSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

const ModelSettingsModal: React.FC<ModelSettingsModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const { mutateAsync, isPending } = useStoreModelInDB();
  const {
    data: proxyConfigData,
    isLoading: isLoadingConfig,
    refetch,
  } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  useEffect(() => {
    if (isVisible) {
      refetch();
    }
  }, [isVisible, refetch]);

  const initialValues = useMemo(() => {
    if (!proxyConfigData) {
      return {
        store_model_in_db: false,
      };
    }

    const storeModelField = proxyConfigData.find(
      (field) => field.field_name === "store_model_in_db",
    );

    return {
      store_model_in_db: storeModelField?.field_value ?? false,
    };
  }, [proxyConfigData]);

  const handleFormSubmit = async (formValues: StoreModelInDBParams) => {
    try {
      await mutateAsync(formValues, {
        onSuccess: () => {
          NotificationsManager.success(
            "Model storage settings updated successfully",
          );
          refetch();
          onSuccess?.();
        },
        onError: (error) => {
          NotificationsManager.fromBackend(
            "Failed to save model storage settings: " + parseErrorMessage(error),
          );
        },
      });
    } catch (error) {
      NotificationsManager.fromBackend(
        "Failed to save model storage settings: " + parseErrorMessage(error),
      );
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  const storeFieldDescription =
    proxyConfigData?.find((f) => f.field_name === "store_model_in_db")
      ?.field_description ||
    "If enabled, models and config are stored in and loaded from the database.";

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">
            Model Settings
          </DialogTitle>
        </DialogHeader>
        <Form
          key={proxyConfigData ? JSON.stringify(initialValues) : "loading"}
          form={form}
          layout="horizontal"
          onFinish={handleFormSubmit}
          initialValues={initialValues}
        >
          <Form.Item
            label="Store Model in DB"
            name="store_model_in_db"
            tooltip={storeFieldDescription}
            valuePropName="checked"
          >
            {isLoadingConfig ? (
              <Skeleton className="h-6 w-full" />
            ) : (
              <AntdFormSwitch />
            )}
          </Form.Item>
        </Form>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={handleCancel}
            disabled={isPending || isLoadingConfig}
          >
            Cancel
          </Button>
          <Button
            onClick={() => form.submit()}
            disabled={isPending || isLoadingConfig}
          >
            {isPending ? "Saving..." : "Save Settings"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ModelSettingsModal;
