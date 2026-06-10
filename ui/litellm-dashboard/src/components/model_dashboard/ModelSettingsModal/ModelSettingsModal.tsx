"use client";

import { ConfigType, useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { StoreModelInDBParams, useStoreModelInDB } from "@/app/(dashboard)/hooks/storeModelInDB/useStoreModelInDB";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { Button, Form, Modal, Skeleton, Space, Switch, Typography } from "antd";
import React, { useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";

interface ModelSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

const ModelSettingsModal: React.FC<ModelSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const { mutateAsync, isPending } = useStoreModelInDB();
  const { data: proxyConfigData, isLoading: isLoadingConfig, refetch } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  // Refetch config when modal opens to ensure we have the latest values
  useEffect(() => {
    if (isVisible) {
      refetch();
    }
  }, [isVisible, refetch]);

  // Compute initial values from fetched config data
  const initialValues = useMemo(() => {
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

  const handleFormSubmit = async (formValues: StoreModelInDBParams) => {
    try {
      await mutateAsync(formValues, {
        onSuccess: () => {
          NotificationsManager.success(t("modelDashboard.modelSettingsModal.updateSuccess"));
          refetch();
          onSuccess?.();
        },
        onError: (error) => {
          NotificationsManager.fromBackend(
            t("modelDashboard.modelSettingsModal.updateFailed", { error: parseErrorMessage(error) }),
          );
        },
      });
    } catch (error) {
      NotificationsManager.fromBackend(
        t("modelDashboard.modelSettingsModal.updateFailed", { error: parseErrorMessage(error) }),
      );
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  return (
    <Modal
      title={<Typography.Title level={5}>{t("modelDashboard.modelSettingsModal.title")}</Typography.Title>}
      open={isVisible}
      footer={
        <Space>
          <Button onClick={handleCancel} disabled={isPending || isLoadingConfig}>
            {t("common.cancel")}
          </Button>
          <Button type="primary" loading={isPending} disabled={isLoadingConfig} onClick={() => form.submit()}>
            {isPending ? t("common.saving") : t("modelDashboard.modelSettingsModal.saveSettings")}
          </Button>
        </Space>
      }
      onCancel={handleCancel}
    >
      <Form
        key={proxyConfigData ? JSON.stringify(initialValues) : "loading"}
        form={form}
        layout="horizontal"
        onFinish={handleFormSubmit}
        initialValues={initialValues}
      >
        <Form.Item
          label={t("modelDashboard.modelSettingsModal.storeModelInDb")}
          name="store_model_in_db"
          tooltip={
            proxyConfigData?.find((f) => f.field_name === "store_model_in_db")?.field_description ||
            t("modelDashboard.modelSettingsModal.storeModelInDbTooltip")
          }
          valuePropName="checked"
        >
          {isLoadingConfig ? <Skeleton.Input active block /> : <Switch />}
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default ModelSettingsModal;
